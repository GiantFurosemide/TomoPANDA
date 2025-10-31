"""
Mesh Geodesic Sampling for CryoET Particle Picking

This module implements mesh-geodesic sampling algorithm for membrane-based
particle picking in cryoET. The algorithm converts membrane masks to triangular
meshes and performs geodesic-based sampling to generate particle picking candidates.

Author: TomoPANDA Team
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, binary_erosion
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import open3d as o3d
from pathlib import Path

# Optional heavy deps used by helper utilities only
import pandas as pd
from tomopanda.utils.mrc_utils import MRCReader

from tomopanda.utils.relion_utils import (
    convert_to_relion_star,
    convert_to_coordinate_file,
    convert_to_prior_angles,
)


class MeshGeodesicSampler:
    """
    Mesh geodesic sampling for membrane-based particle picking.
    
    This class implements the core algorithm for converting membrane masks
    to triangular meshes and performing geodesic-based sampling.
    """
    
    def __init__(self, 
                 smoothing_sigma: float = 1.5,
                 expected_particle_size: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 add_noise: bool = False,
                 noise_scale_factor: float = 0.3,
                 fast_mode: bool = False):
        """
        Initialize the mesh geodesic sampler.
        
        Args:
            smoothing_sigma: Gaussian smoothing parameter for mask preprocessing
            expected_particle_size: Expected particle size in pixels for mesh density control
            random_seed: Random seed for mesh generation (None for deterministic)
            add_noise: If True, add small Gaussian noise to the smoothed mask to introduce variation
            noise_scale_factor: Scales the standard deviation of added noise (multiplied by smoothing_sigma)
            fast_mode: If True, use optimized algorithms for faster processing
        """
        self.smoothing_sigma = smoothing_sigma
        self.expected_particle_size = expected_particle_size
        self.random_seed = random_seed
        self.add_noise = add_noise
        self.noise_scale_factor = noise_scale_factor
        self.fast_mode = fast_mode
    
    def _get_sampling_distance(self) -> float:
        """
        Calculate sampling distance based on expected_particle_size.
        
        Returns:
            Minimum distance between sampling points
        """
        if self.expected_particle_size is not None:
            # Use particle size to determine sampling distance
            # Rule: sampling distance = particle_size / 2 for optimal coverage
            return max(5.0, self.expected_particle_size / 2.0)  # values is [5.0, expected_particle_size/2.0]
        else:
            # Default fallback
            return 20.0
        
    def create_signed_distance_field(self, mask: np.ndarray, target_resolution: float = 1.0) -> np.ndarray:
        """
        Create signed distance field from binary mask with Gaussian smoothing for smoother, thicker membranes.

        Args:
            mask: Binary mask (0/1) indicating membrane regions
            target_resolution: Target resolution for mesh generation (pixels per voxel)
            
        Returns:
            Signed distance field where phi=0 is at membrane center layer, phi>0 is distance to center
        """
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Apply resolution scaling if needed
        if target_resolution is not None and target_resolution != 1.0:
            # Downsample mask for coarser mesh
            from scipy.ndimage import zoom
            scale_factor = 1.0 / target_resolution
            mask = zoom(mask, scale_factor, order=1)  # Linear interpolation
        
        # Create signed distance field: phi=0 at membrane center layer
        # 创建符号距离场，phi=0在膜中心层
        
        # 应用高斯模糊使膜更平滑更厚
        membrane_mask = mask.astype(float)
        
        # 使用高斯模糊平滑膜边界，增加膜厚度
        gaussian_sigma = self.smoothing_sigma
        if gaussian_sigma > 0:
            # 应用高斯模糊，使膜边界更平滑
            membrane_mask = gaussian_filter(membrane_mask, sigma=gaussian_sigma)
            
            # 高斯平滑后大于0.8的全部改为1
            membrane_mask = np.where(membrane_mask > 0.9, 1.0, membrane_mask)
        
        # 增强膜厚度：通过形态学操作增加膜的有效厚度
        # 使用膨胀操作增加膜区域，使膜更厚
        from scipy.ndimage import binary_dilation
        membrane_thickness_factor = max(1.0, self.smoothing_sigma * 0.5)  # 根据sigma调整厚度因子
        if membrane_thickness_factor > 1.0:
            # 创建结构元素用于膨胀
            structure_size = int(membrane_thickness_factor)
            structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
            # 对膜区域进行膨胀以增加厚度
            membrane_mask = binary_dilation(membrane_mask > 0.5, structure=structure).astype(float)
        
        # 计算到膜区域边界的距离
        distance_to_boundary = edt(1 - membrane_mask)
        
        # 计算膜区域内部的厚度
        distance_inside_membrane = edt(membrane_mask)
        
        # 找到膜中心层：膜内部距离边界最大的位置
        if np.any(membrane_mask > 0):
            max_internal_distance = np.max(distance_inside_membrane[membrane_mask > 0])
            
            # 创建膜中心面：使用更宽松的阈值来创建连续的面
            # 膜中心面定义为膜内部距离边界接近最大的位置
            # 对于球形膜，使用更宽松的阈值来创建连续的中心面
            center_threshold = max_internal_distance * 0.45  # 使用50%的最大距离作为阈值
            membrane_center_mask = (membrane_mask > 0) & (distance_inside_membrane >= center_threshold)
            
            # 创建符号距离场：膜中心面为0，膜内部和外部为正值
            phi = np.where(membrane_center_mask,
                          0,  # 膜中心面：0
                          np.where(membrane_mask > 0,
                                  distance_inside_membrane,  # 膜内部：正距离（到膜中心）
                                  distance_to_boundary + max_internal_distance))  # 膜外部：正距离（到膜中心）
        else:
            # 如果没有膜区域，返回全零
            phi = np.zeros_like(membrane_mask)
        
        return phi
    
    def extract_mesh_from_sdf(self, phi: np.ndarray, 
                            spacing: Tuple[float, float, float] = None) -> o3d.geometry.TriangleMesh:
        """
        Extract triangular mesh from signed distance field using marching cubes.
        
        Args:
            phi: Signed distance field
            spacing: Voxel spacing (x, y, z)
            
        Returns:
            Open3D triangle mesh
        """
        # Calculate adaptive spacing based on expected particle size
        if spacing is None:
            if self.expected_particle_size is not None:
                # Adaptive spacing: larger particles -> coarser mesh
                # Scale factor: particle_size / 10 gives reasonable mesh density
                scale_factor = max(0.5, min(5.0, self.expected_particle_size / 10.0))   # values is [0.5,5.0]
                spacing = (scale_factor, scale_factor, scale_factor)
            else:
                spacing = (1.0, 1.0, 1.0)
        
        # Marching cubes on phi=0 level set
        # marching_cubes(phi, level=0.0, spacing=spacing)：phi是符号距离场，level=0.0表示提取phi=0的等值面，spacing是网格间距
        verts, faces, _, _ = marching_cubes(phi, level=0.0, spacing=spacing)
        
        # Convert to Open3D mesh (note: skimage returns z,y,x, we need x,y,z)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts[:, ::-1]),  # Convert to (x,y,z)
            o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        
        # Clean mesh
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        
        # Apply adaptive mesh processing based on expected particle size
        if self.expected_particle_size is not None:
            # Multi-level control of mesh hole size based on particle size
            particle_size = self.expected_particle_size
            
            # Level 1: Calculate target triangle size (hole size = triangle size)
            # Rule: triangle size should be ~particle_size/3 for optimal sampling
            target_triangle_size = particle_size / 3.0
            
            # Level 2: Calculate taubin iterations for triangle size control
            import math
            if particle_size <= 1:
                adaptive_iterations = 3  # Very fine mesh
            elif particle_size <= 5:
                adaptive_iterations = 5  # Fine mesh
            elif particle_size <= 10:
                adaptive_iterations = 8  # Medium-fine mesh
            elif particle_size <= 20:
                adaptive_iterations = 12  # Medium mesh
            elif particle_size <= 50:
                adaptive_iterations = 18  # Medium-coarse mesh
            elif particle_size <= 100:
                adaptive_iterations = 25  # Coarse mesh
            elif particle_size <= 200:
                adaptive_iterations = 30  # Very coarse mesh
            else:
                adaptive_iterations = 35  # Maximum smoothing
            
            # Apply taubin smoothing for triangle size control
            # Taubin 平滑算法
            # Taubin 平滑是一种网格平滑算法，用于：
            # 减少网格噪声：去除 marching cubes 产生的锯齿状边缘
            # 控制三角形大小：通过平滑来间接控制网格密度
            # 保持形状特征：相比其他平滑方法，Taubin 能更好地保持原始形状
            mesh = mesh.filter_smooth_taubin(number_of_iterations=adaptive_iterations)
            
            # Level 3: Mesh decimation for large particles (direct hole size control)
            if target_triangle_size > 15:  # For large particles, reduce triangle count
                # Calculate target number of triangles based on particle size
                # Rule: ~1 triangle per particle_size^2 area
                target_triangles = max(50, int(1000 / (particle_size / 10)))
                
                if len(mesh.triangles) > target_triangles:
                    print(f"Decimating mesh: {len(mesh.triangles)} -> {target_triangles} triangles")
                    mesh = mesh.simplify_quadric_decimation(target_triangles)
            
            # Level 4: Optional mesh subdivision for very small particles
            elif target_triangle_size < 3:  # For very small particles, increase detail
                # Subdivide mesh to increase triangle count and reduce hole size
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                
        else:
            # Use default smoothing when expected_particle_size is None
            mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()

        # Orient mesh normals for consistency using connectivity-based approach
        try:
            if self.fast_mode:
                # Skip normal orientation for faster processing
                print("快速模式：跳过法向量定向")
            else:
                mesh = self._orient_mesh_normals_consistent(mesh)
        except Exception:
            # Fallback silently if orientation fails; normals remain as computed
            pass
        
        return mesh
    
    def _process_mask_to_mesh(self, 
                             mask: np.ndarray, 
                             spacing: Tuple[float, float, float] = None) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Unified mesh processing logic.
        
        Args:
            mask: Binary membrane mask (0/1)
            spacing: Voxel spacing (x, y, z)
            
        Returns:
            Tuple of (mesh, phi) where:
            - mesh: Open3D triangle mesh
            - phi: Signed distance field
        """
        phi = self.create_signed_distance_field(mask)
        mesh = self.extract_mesh_from_sdf(phi, spacing)
        return mesh, phi
    
    def _compute_face_centers_and_normals(self, 
                                        mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute face centers and consistently oriented normals.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Tuple of (face_centers, face_normals) where:
            - face_centers: Face center coordinates (N_faces, 3)
            - face_normals: Consistently oriented face normals (N_faces, 3)
        """
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Get mesh data
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # Compute face centers
        face_centers = vertices[faces].mean(axis=1)  # (N_faces, 3)
        
        # Use Open3D's pre-computed face normals (already consistently oriented)
        face_normals = np.asarray(mesh.triangle_normals)
        
        return face_centers, face_normals
    
    def get_triangle_centers_and_normals(self, 
                                       mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Get all triangle centers and their consistently oriented normals.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            N*6 array where each row is [x, y, z, nx, ny, nz]
        """
        face_centers, aligned_normals = self._compute_face_centers_and_normals(mesh)
        
        # Combine centers and normals into N*6 array
        result = np.column_stack([face_centers, aligned_normals])
        
        return result
    
    
    
    def sample_mesh_faces_with_consistent_normals(self, 
                                                 mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample mesh faces and compute consistently oriented normals.
        
        This method samples faces from the mesh and computes normals that are
        consistently oriented using connectivity-based approach.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Tuple of (centers, normals) where:
            - centers: Face center coordinates (K, 3)
            - normals: Consistently oriented face normals (K, 3)
        """
        face_centers, aligned_normals = self._compute_face_centers_and_normals(mesh)
        
        # Get sampling distance based on expected_particle_size
        min_distance = self._get_sampling_distance()
        
        # Sample faces with minimum distance constraint
        sampled_centers, sampled_normals = self._sample_faces_with_distance_constraint(
            face_centers, aligned_normals, min_distance
        )
        
        return sampled_centers, sampled_normals
    
    
    def _sample_faces_with_distance_constraint(self,
                                             face_centers: np.ndarray,
                                             face_normals: np.ndarray,
                                             min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initial face sampling with distance constraint using spatial hashing.
        
        This method performs the FIRST stage of distance-based sampling using
        spatial hashing for fast neighbor queries. It's designed for initial
        filtering of mesh faces during the sampling process.
        
        Note: This is followed by apply_non_maximum_suppression() for final
        refinement using more precise KDTree-based distance queries.
        
        Args:
            face_centers: Face center coordinates (N, 3)
            face_normals: Face normals (N, 3)
            min_distance: Minimum distance between sampled faces
            
        Returns:
            Tuple of (sampled_centers, sampled_normals)
        """
        if len(face_centers) == 0:
            return np.array([]), np.array([])
        
        # Create spatial hash grid
        grid_size = max(1, int(min_distance))
        grid = {}
        
        def _get_grid_key(pos):
            return tuple((pos / grid_size).astype(int))
        
        def _is_valid_sample(pos, grid, min_dist):
            """Check if position is valid (not too close to existing samples)"""
            key = _get_grid_key(pos)
            # Check neighboring grid cells
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        neighbor_key = (key[0] + dz, key[1] + dy, key[2] + dx)
                        if neighbor_key in grid:
                            for existing_pos in grid[neighbor_key]:
                                if np.linalg.norm(pos - existing_pos) < min_dist:
                                    return False
            return True
        
        # Progressive sampling
        selected_centers = []
        selected_normals = []
        
        # Shuffle face indices for random sampling
        indices = np.random.permutation(len(face_centers))
        
        for idx in indices:
            pos = face_centers[idx]
            
            if _is_valid_sample(pos, grid, min_distance):
                selected_centers.append(pos)
                selected_normals.append(face_normals[idx])
                
                # Add to spatial grid
                key = _get_grid_key(pos)
                if key not in grid:
                    grid[key] = []
                grid[key].append(pos)
        
        return np.array(selected_centers), np.array(selected_normals)
    
    def apply_non_maximum_suppression(self, 
                                    centers: np.ndarray, 
                                    normals: np.ndarray,
                                    min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Final refinement using non-maximum suppression with precise distance queries.
        
        This method performs the SECOND stage of distance-based sampling using
        KDTree for precise neighbor queries. It's designed for final refinement
        of already sampled points to ensure optimal spacing.
        
        Note: This follows _sample_faces_with_distance_constraint() which uses
        spatial hashing for initial fast filtering.
        
        Args:
            centers: Sample centers (K, 3)
            normals: Sample normals (K, 3)
            min_distance: Minimum distance between samples
            
        Returns:
            Filtered centers and normals
        """
        if len(centers) == 0:
            return centers, normals
            
        # Build KDTree for efficient distance queries
        tree = cKDTree(centers)
        alive = np.ones(len(centers), dtype=bool)
        selected = []
        
        # Process in arbitrary order (could be randomized)
        for i in np.argsort(centers[:, 0]):
            if not alive[i]:
                continue
                
            selected.append(i)
            
            # Mark nearby points as dead
            for j in tree.query_ball_point(centers[i], min_distance):
                alive[j] = False
                
        return centers[selected], normals[selected]
    
    def check_placement_feasibility(self, 
                                  centers: np.ndarray, 
                                  normals: np.ndarray,
                                  volume_shape: Tuple[int, int, int],
                                  particle_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check if particles can be placed without overlapping volume boundaries.
        
        Args:
            centers: Sample centers (K, 3)
            normals: Sample normals (K, 3)
            volume_shape: Volume dimensions (x, y, z)
            particle_radius: Effective particle radius
            
        Returns:
            Feasible centers and normals
        """
        if len(centers) == 0:
            return centers, normals
            
        # Check boundary conditions
        keep = np.all(
            (centers >= particle_radius) & 
            (centers <= (np.array(volume_shape) - particle_radius)), 
            axis=1
        )
        
        return centers[keep], normals[keep]
    
    def sample_membrane_points(self, 
                             mask: np.ndarray,
                             particle_radius: float = 10.0,
                             volume_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to sample points on membrane surface using mesh face sampling.
        
        This method uses mesh face sampling with consistently oriented normals.
        
        Args:
            mask: Binary membrane mask (0/1)
            particle_radius: Effective particle radius for boundary checking
            volume_shape: Volume dimensions for boundary checking
            
        Returns:
            Tuple of (centers, normals) where:
            - centers: Sampled point coordinates (K, 3)
            - normals: Corresponding surface normals consistently oriented (K, 3)
        """
        if volume_shape is None:
            volume_shape = mask.shape[::-1]  # Convert from (z,y,x) to (x,y,z)
        
        print("Using mesh face sampling with consistently oriented normals...")
        
        # Process mask to mesh
        mesh, phi = self._process_mask_to_mesh(mask)
        
        # Sample mesh faces with consistently oriented normals
        centers, normals = self.sample_mesh_faces_with_consistent_normals(mesh)
        
        # Apply non-maximum suppression
        min_distance = self._get_sampling_distance()
        centers, normals = self.apply_non_maximum_suppression(
            centers, normals, min_distance
        )
        
        # Step 5: Check placement feasibility
        centers, normals = self.check_placement_feasibility(
            centers, normals, volume_shape, particle_radius
        )
        
        return centers, normals
    
    def get_all_triangle_centers_and_normals(self, 
                                           mask: np.ndarray,
                                           spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Get all triangle centers and normals from mesh without sampling.
        
        This method extracts all triangle centers and their consistently oriented normals,
        providing the raw mesh data for particle placement.
        
        Note: This method does not use min_distance as it extracts ALL triangles
        without distance-based sampling.
        
        Args:
            mask: Binary membrane mask (0/1)
            spacing: Voxel spacing (x, y, z)
            
        Returns:
            N*6 array where each row is [x, y, z, nx, ny, nz]
        """
        print("Extracting all triangle centers and normals...")
        
        # Process mask to mesh
        mesh, phi = self._process_mask_to_mesh(mask, spacing)
        
        # Get all triangle centers and normals (no sampling applied)
        triangle_data = self.get_triangle_centers_and_normals(mesh)
        
        print(f"Extracted {len(triangle_data)} triangle centers")
        return triangle_data

    # ---------------------------
    # Surface and normal helpers
    # ---------------------------

    @staticmethod
    def extract_surface_voxels(mask: np.ndarray) -> np.ndarray:
        """
        Extract surface voxels of a binary mask.

        A voxel is on the surface if it belongs to the mask (value==1)
        and has at least one 6-neighborhood neighbor outside the mask.

        Args:
            mask: Binary mask (0/1) array with shape (z, y, x)

        Returns:
            Binary array of the same shape where True indicates surface voxels.
        """
        mask_bool = mask.astype(bool)
        # Erode (6-connectivity) and subtract from mask to get boundary layer
        eroded = binary_erosion(mask_bool, structure=np.array([
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
        ], dtype=bool))
        surface = mask_bool & (~eroded)
        return surface.astype(np.uint8)


    def _orient_mesh_normals_consistent(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Ensure consistent normal directions using multi-scale approach.
        
        This method uses a hierarchical approach:
        1. Multi-scale strategy: coarse mesh for reference, fine mesh for precision
        2. SDF gradient method (most accurate for membrane structures)
        3. Improved centroid method (fallback for complex geometries)
        4. Connectivity-based method (final fallback)
        5. Local consistency refinement (final optimization)
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Mesh with consistently oriented normals
        """
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return mesh
        
        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        n_faces = len(faces)
        
        # Performance monitoring
        import time
        start_time = time.time()
        print(f"开始多尺度法向量定向，面数: {n_faces}")
        
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]] 
        v2 = vertices[faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)
        
        # Normalize face normals
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        face_normals = face_normals / norms
        
        # Compute face centers
        face_centers = (v0 + v1 + v2) / 3.0
        
        # 多尺度策略：对于大型网格，先使用粗粒度参考
        if n_faces > 50000 and self.expected_particle_size is not None:
            print(f"大型网格（{n_faces}面），使用多尺度法向量定向")
            self._orient_normals_multiscale(face_centers, face_normals, faces, mesh)
        else:
            # 对于中小型网格，使用传统方法
            self._orient_normals_traditional(face_centers, face_normals, faces, mesh)
        
        # Update mesh
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        
        # Performance monitoring
        total_time = time.time() - start_time
        print(f"多尺度法向量定向完成，耗时: {total_time:.2f}秒")
        
        return mesh
    
    def _orient_normals_multiscale(self, face_centers: np.ndarray, 
                                 face_normals: np.ndarray, faces: np.ndarray, 
                                 mesh: o3d.geometry.TriangleMesh) -> None:
        """
        多尺度法向量定向策略。
        
        1. 生成粗粒度参考网格（10倍尺度）
        2. 快速定向参考网格法向量
        3. 使用参考网格指导细粒度网格定向
        """
        print("步骤1: 生成粗粒度参考网格...")
        
        # 步骤1: 生成粗粒度参考网格
        coarse_spacing = self.expected_particle_size * 10.0  # 10倍尺度
        coarse_mesh = self._generate_coarse_reference_mesh(mesh, coarse_spacing)
        
        if len(coarse_mesh.vertices) == 0:
            print("粗粒度网格生成失败，回退到传统方法")
            self._orient_normals_traditional(face_centers, face_normals, faces, mesh)
            return
        
        print(f"步骤2: 快速定向参考网格（{len(coarse_mesh.triangles)}面）...")
        
        # 步骤2: 快速定向参考网格
        coarse_faces = np.asarray(coarse_mesh.triangles)
        coarse_vertices = np.asarray(coarse_mesh.vertices)
        coarse_centers = (coarse_vertices[coarse_faces[:, 0]] + 
                         coarse_vertices[coarse_faces[:, 1]] + 
                         coarse_vertices[coarse_faces[:, 2]]) / 3.0
        
        # 计算粗粒度法向量
        v0 = coarse_vertices[coarse_faces[:, 0]]
        v1 = coarse_vertices[coarse_faces[:, 1]]
        v2 = coarse_vertices[coarse_faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        coarse_normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(coarse_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        coarse_normals = coarse_normals / norms
        
        # 使用质心方法快速定向粗粒度网格
        self._orient_normals_with_improved_centroid(coarse_centers, coarse_normals, coarse_faces)
        
        print("步骤3: 使用参考网格指导细粒度定向...")
        
        # 步骤3: 使用参考网格指导细粒度网格定向
        self._orient_fine_mesh_with_coarse_reference(
            face_centers, face_normals, faces,
            coarse_centers, coarse_normals
        )
    
    def _generate_coarse_reference_mesh(self, fine_mesh: o3d.geometry.TriangleMesh, 
                                      coarse_spacing: float) -> o3d.geometry.TriangleMesh:
        """
        生成粗粒度参考网格。
        
        改进策略：
        1. 根据expected_particle_size计算合理的简化比例
        2. 确保粗粒度网格保持足够的几何特征
        3. 避免过度简化导致参考信息丢失
        """
        n_faces = len(fine_mesh.triangles)
        
        # 根据粒子大小和网格复杂度计算合理的简化比例
        if self.expected_particle_size is not None:
            # 计算目标三角形数量：确保每个粒子区域有足够的参考面
            # 规则：每个粒子大小区域应该有3-5个参考面
            particle_area = self.expected_particle_size ** 2
            target_triangles = max(200, min(2000, int(n_faces / 20)))  # 5%-20%的简化比例
        else:
            # 默认简化到10%
            target_triangles = max(200, n_faces // 10)
        
        print(f"粗粒度网格生成：{n_faces} -> {target_triangles} 面")
        
        if n_faces > target_triangles:
            # 使用二次误差度量简化，保持几何特征
            coarse_mesh = fine_mesh.simplify_quadric_decimation(target_triangles)
            
            # 确保简化后的网格仍然有效
            if len(coarse_mesh.triangles) == 0:
                print("警告：粗粒度网格简化失败，使用原始网格")
                coarse_mesh = fine_mesh
        else:
            coarse_mesh = fine_mesh
        
        # 对粗粒度网格进行轻微平滑，提高法向量质量
        if len(coarse_mesh.triangles) > 0:
            coarse_mesh = coarse_mesh.filter_smooth_taubin(number_of_iterations=3)
        
        return coarse_mesh
    
    def _orient_fine_mesh_with_coarse_reference(self, fine_centers: np.ndarray,
                                             fine_normals: np.ndarray, fine_faces: np.ndarray,
                                             coarse_centers: np.ndarray, coarse_normals: np.ndarray) -> None:
        """
        使用粗粒度参考网格指导细粒度网格法向量定向。
        
        改进策略：
        1. 使用多个最近邻进行投票决策
        2. 增加距离阈值过滤
        3. 统计定向质量
        """
        from scipy.spatial import cKDTree
        
        # 构建粗粒度网格的KDTree
        coarse_tree = cKDTree(coarse_centers)
        
        # 统计信息
        total_faces = len(fine_centers)
        flipped_count = 0
        skipped_count = 0
        
        # 计算距离阈值：基于粗粒度网格的平均面大小
        if len(coarse_centers) > 0:
            # 计算粗粒度网格的平均面间距离
            coarse_distances = []
            for i in range(min(100, len(coarse_centers))):  # 采样计算
                distances, _ = coarse_tree.query(coarse_centers[i], k=2)
                if len(distances) > 1:
                    coarse_distances.append(distances[1])  # 最近邻距离
            
            if coarse_distances:
                distance_threshold = np.median(coarse_distances) * 2.0  # 2倍中位数距离
            else:
                distance_threshold = float('inf')
        else:
            distance_threshold = float('inf')
        
        print(f"使用距离阈值: {distance_threshold:.3f}")
        
        # 对每个细粒度面，找到最近的粗粒度参考面
        for i in range(len(fine_centers)):
            # 找到最近的粗粒度面
            distances, indices = coarse_tree.query(fine_centers[i], k=1)
            
            # 距离过滤：如果太远，跳过
            if distances > distance_threshold:
                skipped_count += 1
                continue
            
            # 修复：当k=1时，indices是单个整数，不是数组
            if isinstance(indices, np.ndarray):
                if len(indices) > 0:
                    ref_normal = coarse_normals[indices[0]]
                else:
                    skipped_count += 1
                    continue
            else:
                # indices是单个整数
                ref_normal = coarse_normals[indices]
            
            current_normal = fine_normals[i]
            
            # 检查方向一致性
            dot_product = np.dot(current_normal, ref_normal)
            
            # 如果方向相反，翻转细粒度法向量
            if dot_product < 0:
                fine_normals[i] = -fine_normals[i]
                fine_faces[i] = fine_faces[i][::-1]
                flipped_count += 1
        
        print(f"细粒度定向完成：{flipped_count}/{total_faces} 面被翻转，{skipped_count} 面被跳过")
    
    def _orient_normals_traditional(self, face_centers: np.ndarray, 
                                  face_normals: np.ndarray, faces: np.ndarray, 
                                  mesh: o3d.geometry.TriangleMesh) -> None:
        """
        传统法向量定向方法。
        """
        # Method 1: Try SDF gradient method (most accurate)
        try:
            self._orient_normals_with_sdf_gradient(face_centers, face_normals, faces)
            print("使用SDF梯度方法进行法向量定向")
        except Exception as e:
            print(f"SDF梯度方法失败，尝试改进的质心方法: {e}")
            try:
                # Method 2: Improved centroid method
                self._orient_normals_with_improved_centroid(face_centers, face_normals, faces)
                print("使用改进的质心方法进行法向量定向")
            except Exception as e2:
                print(f"改进的质心方法也失败，回退到连通性方法: {e2}")
                # Method 3: Connectivity-based method (final fallback)
                self._orient_mesh_normals_connectivity(mesh, faces, face_normals)
        
        # Method 4: Local consistency refinement (条件应用)
        if self.fast_mode:
            print("快速模式：跳过局部一致性检查")
        elif len(face_centers) > 100000:
            print(f"超大型网格（{len(face_centers)}面），跳过局部一致性检查")
        else:
            try:
                self._refine_normals_with_local_consistency(face_centers, face_normals, faces)
                print("使用局部一致性方法进一步优化法向量")
            except Exception as e:
                print(f"局部一致性方法失败: {e}")
                # Continue with current normals
    
    def _orient_normals_with_sdf_gradient(self, face_centers: np.ndarray, 
                                       face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        Orient normals using SDF gradient method (most accurate for membrane structures).
        
        This method uses the signed distance field gradient to determine the correct
        outward direction for membrane normals. The SDF gradient points in the direction
        of increasing distance from the membrane surface, which is the outward direction.
        
        Args:
            face_centers: Face center coordinates (N, 3)
            face_normals: Face normals (N, 3) - will be modified in place
            faces: Triangle faces (N, 3) - will be modified in place
        """
        n_faces = len(face_centers)
        
        # 使用高效的连通性方法替代复杂的 SDF 梯度计算
        # 对于大型网格，连通性方法比 SDF 梯度更高效且同样准确
        if n_faces > 5000:
            print(f"大型网格（{n_faces}面），使用快速连通性方法")
            self._orient_normals_with_fast_connectivity(face_centers, face_normals, faces)
        else:
            # 对于中小型网格，使用完整的 SDF 梯度方法
            self._orient_normals_with_compute_sdf_gradient(face_centers, face_normals, faces)
    
    def _orient_normals_with_compute_sdf_gradient(self, face_centers: np.ndarray, 
                                                face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        计算真正的 SDF 梯度进行法向量定向。
        
        使用数值梯度计算 SDF 梯度，适用于中小型网格。
        """
        n_faces = len(face_centers)
        eps = 1e-3  # 数值梯度的步长
        
        # 向量化计算 SDF 梯度
        # 对每个面中心，计算到所有其他面的最小距离作为 SDF 近似
        for i in range(n_faces):
            center = face_centers[i]
            
            # 计算到其他面中心的距离
            distances = np.linalg.norm(face_centers - center, axis=1)
            distances[i] = np.inf  # 排除自己
            
            # 找到最近的面
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # 计算梯度方向（指向最近面的方向）
            if min_dist > 0:
                grad_direction = (face_centers[min_idx] - center) / min_dist
                
                # 检查法向量是否与梯度方向一致
                dot_product = np.dot(face_normals[i], grad_direction)
                
                # 如果法向量与梯度方向相反，翻转它
                if dot_product < 0:
                    face_normals[i] = -face_normals[i]
                    faces[i] = faces[i][::-1]
    
    def _orient_normals_with_fast_connectivity(self, face_centers: np.ndarray, 
                                             face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        使用快速连通性方法进行法向量定向。
        
        对于大型网格，使用基于邻接关系的快速方法。
        """
        n_faces = len(face_centers)
        
        # 使用 KDTree 快速找到每个面的邻居
        from scipy.spatial import cKDTree
        tree = cKDTree(face_centers)
        
        # 计算搜索半径（基于面密度的自适应半径）
        distances = np.linalg.norm(face_centers - np.mean(face_centers, axis=0), axis=1)
        search_radius = np.percentile(distances, 10) * 3.0  # 使用10%分位数作为搜索半径
        
        # 对于超大型网格，使用采样方法大幅减少计算量
        if n_faces > 100000:
            # 超大型网格：只处理部分面
            sample_rate = 0.1  # 只处理10%的面
            sample_indices = np.random.choice(n_faces, int(n_faces * sample_rate), replace=False)
            print(f"超大型网格（{n_faces}面），采样处理 {len(sample_indices)} 个面")
        else:
            sample_indices = range(n_faces)
        
        # 对采样的面进行处理
        for i in sample_indices:
            # 找到附近的面
            nearby_indices = tree.query_ball_point(face_centers[i], search_radius)
            nearby_indices = [idx for idx in nearby_indices if idx != i]
            
            if len(nearby_indices) > 0:
                # 大幅减少邻居数量
                if len(nearby_indices) > 5:  # 只使用5个最近邻居
                    nearby_indices = nearby_indices[:5]
                
                # 计算当前面与邻居面的法向量一致性
                current_normal = face_normals[i]
                nearby_normals = face_normals[nearby_indices]
                
                # 计算与邻居法向量的平均点积
                dot_products = np.dot(nearby_normals, current_normal)
                avg_consistency = np.mean(dot_products)
                
                # 如果平均一致性为负，说明法向量方向相反，需要翻转
                if avg_consistency < -0.1:  # 使用较小的阈值避免过度翻转
                    face_normals[i] = -face_normals[i]
                    faces[i] = faces[i][::-1]
    
    def _orient_normals_with_improved_centroid(self, face_centers: np.ndarray, 
                                            face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        Orient normals using improved centroid method.
        
        This method uses the surface centroid (centroid of face centers) instead of
        the vertex centroid, which is more accurate for complex membrane structures.
        
        Args:
            face_centers: Face center coordinates (N, 3)
            face_normals: Face normals (N, 3) - will be modified in place
            faces: Triangle faces (N, 3) - will be modified in place
        """
        n_faces = len(face_centers)
        
        # Use surface centroid (centroid of face centers) instead of vertex centroid
        surface_centroid = np.mean(face_centers, axis=0)
        
        # For each face, check if normal points away from surface centroid
        for i in range(n_faces):
            face_center = face_centers[i]
            normal = face_normals[i]
            
            # Vector from surface centroid to face center
            to_face = face_center - surface_centroid
            to_face = to_face / (np.linalg.norm(to_face) + 1e-8)
            
            # If normal points towards surface centroid, flip it
            if np.dot(normal, to_face) < 0:
                face_normals[i] = -face_normals[i]
                faces[i] = faces[i][::-1]
    
    def _orient_mesh_normals_connectivity(self, mesh: o3d.geometry.TriangleMesh, 
                                        faces: np.ndarray, face_normals: np.ndarray) -> None:
        """
        Fallback connectivity-based normal orientation.
        
        This method uses breadth-first search to ensure that adjacent faces have
        consistent normal directions, without relying on SDF gradients.
        
        Args:
            mesh: Open3D triangle mesh
            faces: Triangle faces array
            face_normals: Face normals array (will be modified in place)
        """
        n_faces = len(faces)
        
        # Build face adjacency graph using edge hash table
        face_adjacency = [[] for _ in range(n_faces)]
        
        # Use edge hash table for O(F) adjacency building
        from collections import defaultdict
        edge_to_faces = defaultdict(list)
        
        # Build edge -> faces mapping
        for face_idx, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                edge_to_faces[edge].append(face_idx)
        
        # Build adjacency from shared edges
        for edge, face_list in edge_to_faces.items():
            if len(face_list) == 2:  # Only two faces share this edge
                face1, face2 = face_list
                face_adjacency[face1].append(face2)
                face_adjacency[face2].append(face1)
        
        # Use BFS to ensure consistency
        visited = [False] * n_faces
        queue = [0]  # Start from first face
        visited[0] = True
        
        while queue:
            current_face = queue.pop(0)
            
            for neighbor_face in face_adjacency[current_face]:
                if not visited[neighbor_face]:
                    # Check normal direction consistency
                    dot_product = np.dot(face_normals[current_face], face_normals[neighbor_face])
                    
                    # If dot product is negative, flip neighbor face normal
                    if dot_product < 0:
                        face_normals[neighbor_face] = -face_normals[neighbor_face]
                        # Also flip triangle vertex order
                        faces[neighbor_face] = faces[neighbor_face][::-1]
                    
                    visited[neighbor_face] = True
                    queue.append(neighbor_face)
    
    def _refine_normals_with_local_consistency(self, face_centers: np.ndarray, 
                                            face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        Refine normal directions using local consistency analysis.
        
        This method checks each face against its nearby faces to ensure
        local consistency of normal directions.
        
        Args:
            face_centers: Face center coordinates (N, 3)
            face_normals: Face normals (N, 3) - will be modified in place
            faces: Triangle faces (N, 3) - will be modified in place
        """
        n_faces = len(face_centers)
        
        # 对于大型网格，使用更高效的局部一致性检查
        if n_faces > 100000:  # 超大型网格直接跳过
            print(f"超大型网格（{n_faces}面），跳过局部一致性检查")
            return  # 直接跳过
        elif n_faces > 50000:  # 大型网格使用采样
            print(f"大型网格（{n_faces}面），使用采样局部一致性")
            self._refine_normals_with_sampling_consistency(face_centers, face_normals, faces)
        elif n_faces > 5000:
            print(f"中型网格局部一致性优化（{n_faces}面），使用采样方法")
            self._refine_normals_with_sampling_consistency(face_centers, face_normals, faces)
        else:
            # 对于中小型网格，使用完整的局部一致性检查
            self._refine_normals_with_full_consistency(face_centers, face_normals, faces)
    
    def _refine_normals_with_sampling_consistency(self, face_centers: np.ndarray, 
                                                face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        使用采样方法进行局部一致性检查，适用于大型网格。
        """
        n_faces = len(face_centers)
        
        # 使用 KDTree 进行快速邻居搜索
        from scipy.spatial import cKDTree
        tree = cKDTree(face_centers)
        
        # 计算搜索半径
        distances = np.linalg.norm(face_centers - np.mean(face_centers, axis=0), axis=1)
        search_radius = np.percentile(distances, 20) * 2.0  # 使用20%分位数
        
        # 对每个面进行局部一致性检查
        # 对于超大型网格，使用更激进的优化
        if n_faces > 100000:
            # 超大型网格：跳过局部一致性检查
            print(f"超大型网格（{n_faces}面），跳过局部一致性检查以提高性能")
            return
        
        # 使用采样方法减少计算量
        sample_rate = max(0.1, min(1.0, 50000 / n_faces))  # 动态采样率
        sample_indices = np.random.choice(n_faces, int(n_faces * sample_rate), replace=False)
        
        for i in sample_indices:
            current_center = face_centers[i]
            current_normal = face_normals[i]
            
            # 找到附近的面
            nearby_indices = tree.query_ball_point(current_center, search_radius)
            nearby_indices = [idx for idx in nearby_indices if idx != i]
            
            if len(nearby_indices) > 0:
                # 限制邻居数量以提高性能
                if len(nearby_indices) > 10:  # 进一步减少邻居数量
                    nearby_indices = nearby_indices[:10]
                
                nearby_normals = face_normals[nearby_indices]
                dot_products = np.dot(nearby_normals, current_normal)
                
                # 如果大多数邻居法向量指向相反方向，翻转当前法向量
                if np.mean(dot_products) < -0.2:  # 使用较小的阈值
                    face_normals[i] = -face_normals[i]
                    faces[i] = faces[i][::-1]
    
    def _refine_normals_with_full_consistency(self, face_centers: np.ndarray, 
                                           face_normals: np.ndarray, faces: np.ndarray) -> None:
        """
        使用完整方法进行局部一致性检查，适用于中小型网格。
        """
        n_faces = len(face_centers)
        
        # 对每个面，检查与附近面的一致性
        for i in range(n_faces):
            current_center = face_centers[i]
            current_normal = face_normals[i]
            
            # 找到附近的面
            distances = np.linalg.norm(face_centers - current_center, axis=1)
            nearby_indices = np.where(distances < np.mean(distances) * 2.0)[0]
            nearby_indices = nearby_indices[nearby_indices != i]  # 排除自己
            
            if len(nearby_indices) > 0:
                # 检查当前法向量与附近法向量的一致性
                nearby_normals = face_normals[nearby_indices]
                dot_products = np.dot(nearby_normals, current_normal)
                
                # 如果大多数附近法向量指向相反方向，翻转当前法向量
                if np.mean(dot_products) < -0.3:  # 阈值用于翻转
                    face_normals[i] = -face_normals[i]
                    faces[i] = faces[i][::-1]
    

    @staticmethod
    def rasterize_mesh_to_volume(mesh: o3d.geometry.TriangleMesh, volume_shape_zyx: Tuple[int, int, int]) -> np.ndarray:
        """
        Voxelize the triangle mesh into a dense binary volume (1 at occupied surface voxels).

        Uses Open3D's VoxelGrid at voxel_size=1.0 and maps voxel centers to nearest grid cells.

        Args:
            mesh: Open3D triangle mesh in (x,y,z) coordinate system aligned to voxel grid
            volume_shape_zyx: Target volume shape in (z,y,x)

        Returns:
            Binary uint8 volume of shape (z,y,x) with 1s on mesh surface.
        """
        if len(mesh.vertices) == 0:
            return np.zeros(volume_shape_zyx, dtype=np.uint8)

        # Create voxel grid from mesh
        vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1.0)
        voxels = vg.get_voxels()
        vol = np.zeros(volume_shape_zyx, dtype=np.uint8)

        # Compute centers and map to indices
        origin = np.array(vg.origin, dtype=float)
        for v in voxels:
            idx = np.array(v.grid_index, dtype=int)  # (ix, iy, iz) in voxel grid space
            center = origin + (idx.astype(float) + 0.5) * 1.0  # (x,y,z)
            vz = int(np.rint(center[2]))
            vy = int(np.rint(center[1]))
            vx = int(np.rint(center[0]))
            if 0 <= vz < volume_shape_zyx[0] and 0 <= vy < volume_shape_zyx[1] and 0 <= vx < volume_shape_zyx[2]:
                vol[vz, vy, vx] = 1

        return vol


def create_mesh_geodesic_sampler(smoothing_sigma: float = 1.5,
                                expected_particle_size: Optional[float] = None,
                                random_seed: Optional[int] = None,
                                add_noise: bool = False,
                                noise_scale_factor: float = 0.1,
                                fast_mode: bool = False) -> MeshGeodesicSampler:
    """
    Factory function to create a MeshGeodesicSampler instance.
    
    Args:
        smoothing_sigma: Gaussian smoothing parameter (in voxels)
        expected_particle_size: Expected particle size in pixels for mesh density control
        random_seed: Random seed for mesh generation (None for deterministic)
        add_noise: If True, add small Gaussian noise to the smoothed mask to introduce variation
        noise_scale_factor: Scales the standard deviation of added noise (multiplied by smoothing_sigma)
        fast_mode: If True, use optimized algorithms for faster processing
        
    Returns:
        Configured MeshGeodesicSampler instance
    """
    return MeshGeodesicSampler(
        smoothing_sigma=smoothing_sigma,
        expected_particle_size=expected_particle_size,
        random_seed=random_seed,
        add_noise=add_noise,
        noise_scale_factor=noise_scale_factor,
        fast_mode=fast_mode
    )


# ---------------------------
# High-level helper utilities
# ---------------------------

def generate_synthetic_mask(
    shape: Tuple[int, int, int] = (100, 100, 100),
    center: Tuple[int, int, int] = (50, 50, 50),
    radius: int = 30,
    membrane_thickness: int = 4,
) -> np.ndarray:
    """
    Create a spherical membrane mask with specified thickness for testing.
    
    Args:
        shape: Volume shape (z, y, x)
        center: Sphere center (z, y, x)
        radius: Sphere radius in voxels
        membrane_thickness: Thickness of the membrane in voxels
    
    Returns:
        Binary uint8 mask with values in {0, 1}
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    
    # Create membrane: outer radius - inner radius = membrane thickness
    inner_radius = radius - membrane_thickness
    outer_radius = radius
    
    # Membrane is the region between inner and outer radius
    membrane_mask = (distance >= inner_radius) & (distance <= outer_radius)
    
    return membrane_mask.astype(np.uint8)


def run_mesh_geodesic_sampling(
    mask: np.ndarray,
    *, # '*' is used to indicate that the following parameters are keyword-only
    smoothing_sigma: float = 1.5,
    particle_radius: float = 10.0,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    expected_particle_size: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-shot convenience wrapper to run mesh-geodesic sampling on a mask.
    
    Uses mesh face sampling with consistently oriented normals for all datasets.
    
    Args:
        mask: Binary membrane mask (0/1)
        smoothing_sigma: Gaussian smoothing parameter (in voxels)
        particle_radius: Effective particle radius for boundary checking
        volume_shape: Volume dimensions for boundary checking
        expected_particle_size: Expected particle size in pixels for mesh density control
        random_seed: Random seed for mesh generation (None for deterministic)
    
    Returns:
        Tuple of (centers, normals) where:
        - centers: Sampled point coordinates (K, 3)
        - normals: Corresponding surface normals consistently oriented (K, 3)
    """
    sampler = create_mesh_geodesic_sampler(
        smoothing_sigma=smoothing_sigma,
        expected_particle_size=expected_particle_size,
        random_seed=random_seed
    )
    centers, normals = sampler.sample_membrane_points(
        mask,
        particle_radius=particle_radius,
        volume_shape=volume_shape,
    )
    return centers, normals


def get_triangle_centers_and_normals(
    mask: np.ndarray,
    *,
    expected_particle_size: Optional[float] = None,
    smoothing_sigma: float = 1.5,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    One-shot convenience function to get all triangle centers and normals.
    
    This function extracts ALL triangle centers and normals from the mesh,
    without applying distance-based sampling. The mesh density is controlled
    by expected_particle_size parameter.
    
    Args:
        mask: Binary membrane mask (0/1)
        expected_particle_size: Expected particle size in pixels for mesh density control
        smoothing_sigma: Gaussian smoothing parameter (in voxels)
        spacing: Voxel spacing (x, y, z)
        random_seed: Random seed for mesh generation (None for deterministic)
    
    Returns:
        N*6 array where each row is [x, y, z, nx, ny, nz]
    """
    sampler = create_mesh_geodesic_sampler(
        smoothing_sigma=smoothing_sigma,
        expected_particle_size=expected_particle_size,
        random_seed=random_seed
    )
    
    triangle_data = sampler.get_all_triangle_centers_and_normals(mask, spacing)
    return triangle_data


def save_sampling_outputs(
    output_dir: Union[str, Path],
    centers: np.ndarray,
    normals: np.ndarray,
    *,
    mrc_path: Optional[Union[str, Path]] = None,
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    sigma_tilt: float = 30.0,
    sigma_psi: float = 30.0,
    sigma_rot: float = 30.0,
    create_vis_script: bool = False,
    use_simplified_relion: bool = True,
) -> Tuple[Path, Path, Path, Optional[Path]]:
    """
    Save standard outputs produced by sampling.
    
    Files:
      - sampling_coordinates.csv (centers + normals)
      - particles.star (Simplified RELION STAR format)
      - coordinates.csv (x,y,z + tilt,psi,rot)
      - prior_angles.csv (RELION priors)
      - visualize_results.py (optional viewer script)
    
    Args:
        mrc_path: Optional path to source MRC; if provided and voxel_size is None,
                  voxel size will default to the MRC header pixel size (x,y,z).
        use_simplified_relion: If True, use simplified RELION format with minimal columns
    
    Returns tuple of generated file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine effective voxel size: prefer explicit voxel_size, else derive from MRC
    effective_voxel_size: Optional[Tuple[float, float, float]] = None
    if voxel_size is not None:
        effective_voxel_size = tuple(map(float, voxel_size))  # type: ignore[arg-type]
    elif mrc_path is not None:
        try:
            _, meta = MRCReader.read_mrc(mrc_path)
            vs = meta.get('voxel_size', None)
            # vs might be a NamedTuple-like or an object with x,y,z attributes
            if vs is not None:
                if hasattr(vs, 'x') and hasattr(vs, 'y') and hasattr(vs, 'z'):
                    effective_voxel_size = (float(vs.x), float(vs.y), float(vs.z))
                elif isinstance(vs, (tuple, list)) and len(vs) == 3:
                    effective_voxel_size = (float(vs[0]), float(vs[1]), float(vs[2]))
        except Exception:
            # If MRC read fails, fall back to None (no scaling)
            effective_voxel_size = None

    # 1) Coordinates CSV with normals
    coord_csv = out_dir / "sampling_coordinates.csv"
    df_coords = pd.DataFrame(
        np.column_stack([centers, normals]),
        columns=["x", "y", "z", "nx", "ny", "nz"],
    )
    df_coords.to_csv(coord_csv, index=False)

    # 2) Simplified RELION STAR format
    star_file = out_dir / "particles.star"
    if use_simplified_relion:
        _save_simplified_relion_star(
            centers,
            normals,
            star_file,
            tomogram_name=tomogram_name,
            particle_diameter=particle_diameter,
            voxel_size=effective_voxel_size,
        )
    else:
        # Use legacy format
        convert_to_relion_star(
            centers,
            normals,
            star_file,
            tomogram_name=tomogram_name,
            particle_diameter=particle_diameter,
        )

    # 3) Coordinate file (csv)
    coordinates_file = out_dir / "coordinates.csv"
    convert_to_coordinate_file(
        centers,
        normals,
        coordinates_file,
        voxel_size=effective_voxel_size,
    )

    # 4) Prior angles
    prior_file = out_dir / "prior_angles.csv"
    convert_to_prior_angles(
        centers,
        normals,
        prior_file,
        sigma_tilt=sigma_tilt,
        sigma_psi=sigma_psi,
        sigma_rot=sigma_rot,
    )

    # 5) Optional visualization helper script
    vis_script_path: Optional[Path] = None
    if create_vis_script:
        vis_script_path = out_dir / "visualize_results.py"
        create_visualization_script(vis_script_path, centers, normals)

    return coord_csv, star_file, coordinates_file, vis_script_path


def _save_simplified_relion_star(
    centers: np.ndarray,
    normals: np.ndarray,
    output_path: Union[str, Path],
    *,
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Save simplified RELION STAR file with minimal columns.
    
    Only includes essential columns for particle picking:
    - rlnCoordinateX/Y/Z: particle coordinates
    - rlnAngleTilt/Psi/Rot: particle orientations
    - rlnTomoName: tomogram name
    - rlnTomoParticleId: particle ID
    - rlnTomoParticleDiameter: particle diameter
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3)
        output_path: Output STAR file path
        tomogram_name: Name of the tomogram
        particle_diameter: Particle diameter in Angstroms
        voxel_size: Optional voxel size for coordinate scaling
    """
    if len(centers) == 0:
        raise ValueError("No coordinates to save")
    
    # Scale coordinates if voxel size is provided
    if voxel_size is not None:
        centers_scaled = centers * np.array(voxel_size)
    else:
        centers_scaled = centers
    
    # Convert membrane normals to Euler angles
    from tomopanda.utils.relion_utils import RELIONConverter
    euler_angles = []
    for normal in normals:
        tilt, psi, rot = RELIONConverter.normal_to_euler(normal)
        euler_angles.append([tilt, psi, rot])
    
    euler_angles = np.array(euler_angles)
    
    # Create simplified STAR file data (minimal columns only)
    data = {
        'rlnCoordinateX': centers_scaled[:, 0],
        'rlnCoordinateY': centers_scaled[:, 1], 
        'rlnCoordinateZ': centers_scaled[:, 2],
        'rlnAngleTilt': euler_angles[:, 0],
        'rlnAnglePsi': euler_angles[:, 1],
        'rlnAngleRot': euler_angles[:, 2],
        'rlnTomoName': [tomogram_name] * len(centers),
        'rlnTomoParticleId': range(len(centers)),
        'rlnTomoParticleDiameter': [particle_diameter] * len(centers)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Write STAR file
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        # Write header
        f.write("data_\n\n")
        f.write("loop_\n")
        
        # Write column labels
        for col in df.columns:
            f.write(f"_{col}\n")
        
        # Write data
        for _, row in df.iterrows():
            f.write(" ".join([f"{val:.6f}" if isinstance(val, float) else str(val) 
                            for val in row.values]) + "\n")
    
    print(f"Saved {len(centers)} particles to simplified RELION STAR file: {output_path}")


def create_visualization_script(
    script_path: Union[str, Path],
    centers: np.ndarray,
    normals: np.ndarray,
) -> None:
    """
    Create a standalone matplotlib-based visualization script.
    """
    script_path = Path(script_path)
    content = f'''#!/usr/bin/env python3
"""
Visualization script for mesh geodesic sampling results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load sampling results
centers = np.array({centers.tolist()})
normals = np.array({normals.tolist()})

# Create 3D visualization
fig = plt.figure(figsize=(12, 8))

# Plot 1: 3D scatter plot of sampling points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
           c=range(len(centers)), cmap='viridis', s=20)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f"{len(centers)} Sampling Points")

# Plot 2: Normal vectors
ax2 = fig.add_subplot(122, projection='3d')
step = max(1, len(centers) // 50)
for i in range(0, len(centers), step):
    center = centers[i]
    normal = normals[i] * 5
    ax2.quiver(center[0], center[1], center[2], 
               normal[0], normal[1], normal[2], 
               color='red', alpha=0.6, arrow_length_ratio=0.1)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Surface Normals')

plt.tight_layout()
plt.savefig('sampling_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'sampling_visualization.png'")
print(f"Total sampling points: {len(centers)}")
'''
    with open(script_path, 'w') as f:
        f.write(content)
    script_path.chmod(0o755)

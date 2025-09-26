"""
Mesh Geodesic Sampling for CryoET Particle Picking

This module implements mesh-geodesic sampling algorithm for membrane-based
particle picking in cryoET. The algorithm converts membrane masks to triangular
meshes and performs geodesic-based sampling to generate particle picking candidates.

Author: TomoPANDA Team
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, binary_erosion
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import marching_cubes
import open3d as o3d
from pathlib import Path

# Optional heavy deps used by helper utilities only
import pandas as pd

from tomopanda.utils.relion_utils import (
    convert_to_relion_star,
    convert_to_coordinate_file,
    convert_to_prior_angles,
)
from tomopanda.utils.mrc_utils import MRCWriter


class MeshGeodesicSampler:
    """
    Mesh geodesic sampling for membrane-based particle picking.
    
    This class implements the core algorithm for converting membrane masks
    to triangular meshes and performing geodesic-based sampling.
    """
    
    def __init__(self, 
                 smoothing_sigma: float = 1.5,
                 taubin_iterations: int = 10,
                 min_distance: float = 20.0):
        """
        Initialize the mesh geodesic sampler.
        
        Args:
            smoothing_sigma: Gaussian smoothing parameter for mask preprocessing
            taubin_iterations: Number of Taubin smoothing iterations
            min_distance: Minimum distance between sampling points
        """
        self.smoothing_sigma = smoothing_sigma
        self.taubin_iterations = taubin_iterations
        self.min_distance = min_distance
        
    def create_signed_distance_field(self, mask: np.ndarray) -> np.ndarray:
        """
        Create signed distance field from binary mask.
        
        Args:
            mask: Binary mask (0/1) indicating membrane regions
            
        Returns:
            Signed distance field where phi>0 is outside membrane, phi<0 is inside
        """
        # Apply Gaussian smoothing to reduce noise
        mask_smooth = gaussian_filter(mask.astype(float), sigma=self.smoothing_sigma)
        
        # Create signed distance field: outside - inside
        phi = edt(1 - (mask_smooth > 0.5)) - edt((mask_smooth > 0.5))
        
        return phi
    
    def extract_mesh_from_sdf(self, phi: np.ndarray, 
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> o3d.geometry.TriangleMesh:
        """
        Extract triangular mesh from signed distance field using marching cubes.
        
        Args:
            phi: Signed distance field
            spacing: Voxel spacing (x, y, z)
            
        Returns:
            Open3D triangle mesh
        """
        # Marching cubes on phi=0 level set
        verts, faces, _, _ = marching_cubes(phi, level=0.0, spacing=spacing)
        
        # Convert to Open3D mesh (note: skimage returns z,y,x, we need x,y,z)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts[:, ::-1]),  # Convert to (x,y,z)
            o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        
        # Clean mesh
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        
        # Apply Taubin smoothing
        mesh = mesh.filter_smooth_taubin(number_of_iterations=self.taubin_iterations)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()

        # Try to orient mesh normals outward using the SDF gradient
        try:
            mesh = self._orient_mesh_normals_outward(mesh, phi)
        except Exception:
            # Fallback silently if orientation fails; normals remain as computed
            pass
        
        return mesh
    
    def geodesic_farthest_point_sampling(self, 
                                       vertices: np.ndarray, 
                                       faces: np.ndarray,
                                       radius: float,
                                       upper_limit: int = 100000000) -> np.ndarray:
        """
        [DEPRECATED] Perform geodesic farthest point sampling on mesh.
        
        This method is too slow for large datasets (671x671x350).
        Replaced by fast_voxel_based_sampling for better performance.
        
        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3), each face is a tuple of 3 vertex indices
            radius: Minimum geodesic distance between samples
            
        Returns:
            Array of vertex indices for sampled points
        """
        # This method is commented out due to performance issues with large datasets
        # Use fast_voxel_based_sampling instead for 671x671x350 level data
        raise NotImplementedError(
            "Geodesic sampling is too slow for large datasets. "
            "Use fast_voxel_based_sampling instead."
        )
        
        # # Build mesh edge graph once (used if gdist is unavailable)
        # def _build_mesh_graph(verts: np.ndarray, tris: np.ndarray) -> csr_matrix:
        #     i_list = []
        #     j_list = []
        #     w_list = []
        #     # Each triangle contributes three undirected edges
        #     for f in tris:
        #         a, b, c = int(f[0]), int(f[1]), int(f[2])
        #         for u, v in ((a, b), (b, c), (c, a)):
        #             if u == v:
        #                 continue
        #             duv = np.linalg.norm(verts[u] - verts[v])
        #             i_list.extend([u, v])
        #             j_list.extend([v, u])
        #             w_list.extend([duv, duv])
        #     n = len(verts)
        #     graph = coo_matrix((w_list, (i_list, j_list)), shape=(n, n))
        #     return graph.tocsr()

        # # Try to use gdist if available (fast and accurate). Otherwise fall back to graph shortest paths.
        # try:
        #     import gdist  # type: ignore
        #     use_gdist = True
        # except Exception:
        #     use_gdist = False
        #     graph_csr = _build_mesh_graph(vertices, faces)
        
        # picked = []
        # n_vertices = len(vertices)
        
        # # Start with vertex closest to centroid
        # centroid = vertices.mean(0)
        # start_idx = np.argmin(np.linalg.norm(vertices - centroid, axis=1))
        # picked.append(start_idx)
        
        # # Maintain minimum geodesic distance to picked set
        # dmin = np.full(n_vertices, np.inf)
        
        # for _ in range(upper_limit):  # Upper limit, will break early
        #     # Compute geodesic distances from latest seed point
        #     if use_gdist:
        #         d = gdist.compute_gdist(
        #             vertices.astype(np.float64), 
        #             faces, 
        #             np.array([picked[-1]], dtype=np.int32)
        #         )
        #     else:
        #         # Single-source shortest paths on the edge-weighted mesh graph
        #         d = dijkstra(graph_csr, directed=False, indices=picked[-1])
        #     dmin = np.minimum(dmin, d)
            
        #     # Select vertex with maximum minimum distance
        #     next_idx = int(np.argmax(dmin))
            
        #     if dmin[next_idx] < radius:
        #         # All vertices are within radius of picked set
        #         break
                
        #     picked.append(next_idx)
            
        # return np.array(picked, dtype=int)
    
    # def fast_voxel_based_sampling(self, 
    #                             mask: np.ndarray,
    #                             min_distance: float = 20.0,
    #                             target_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Fast voxel-based sampling for large datasets (671x671x350).
    #     
    #     This method uses a multi-scale approach:
    #     1. Extract surface voxels from mask
    #     2. Use spatial hashing for fast distance queries
    #     3. Progressive sampling with grid-based rejection
    #     
    #     Args:
    #         mask: Binary mask (0/1) indicating membrane regions
    #         min_distance: Minimum distance between samples (in voxels)
    #         target_samples: Target number of samples to generate
    #         
    #     Returns:
    #         Tuple of (centers, normals) where:
    #         - centers: Sampled point coordinates (K, 3)
    #         - normals: Corresponding surface normals (K, 3)
    #     """
    #     print(f"Starting fast voxel-based sampling for {mask.shape} mask...")
    #     
    #     # Step 1: Extract surface voxels efficiently
    #     surface_voxels = self._extract_surface_voxels_fast(mask)
    #     if len(surface_voxels) == 0:
    #         return np.array([]), np.array([])
    #     
    #     print(f"Found {len(surface_voxels)} surface voxels")
    #     
    #     # Step 2: Compute surface normals using gradient
    #     normals = self._compute_surface_normals_fast(mask, surface_voxels)
    #     
    #     # Step 3: Fast spatial sampling with grid-based rejection
    #     centers, normals = self._fast_spatial_sampling(
    #         surface_voxels, normals, min_distance, target_samples
    #     )
    #     
    #     print(f"Generated {len(centers)} samples")
    #     return centers, normals
    
    # def _extract_surface_voxels_fast(self, mask: np.ndarray) -> np.ndarray:
    #     """
    #     Fast extraction of surface voxels using morphological operations.
    #     """
    #     # Use 6-connectivity structure for 3D erosion
    #     structure = np.array([
    #         [[0, 0, 0],
    #          [0, 1, 0],
    #          [0, 0, 0]],
    #         [[0, 1, 0],
    #          [1, 1, 1],
    #          [0, 1, 0]],
    #         [[0, 0, 0],
    #          [0, 1, 0],
    #          [0, 0, 0]],
    #     ], dtype=bool)
    #     
    #     # Erode and subtract to get boundary
    #     eroded = binary_erosion(mask.astype(bool), structure=structure)
    #     surface_mask = mask.astype(bool) & (~eroded)
    #     
    #     # Get coordinates of surface voxels
    #     surface_coords = np.argwhere(surface_mask)
    #     return surface_coords.astype(np.float32)
    
    # def _compute_surface_normals_fast(self, mask: np.ndarray, surface_voxels: np.ndarray) -> np.ndarray:
    #     """
    #     Fast computation of surface normals using finite differences.
    #     """
    #     # Compute gradients using finite differences
    #     grad_z, grad_y, grad_x = np.gradient(mask.astype(np.float32))
    #     
    #     # Sample gradients at surface voxel locations
    #     z_coords = np.clip(surface_voxels[:, 0].astype(int), 0, mask.shape[0]-1)
    #     y_coords = np.clip(surface_voxels[:, 1].astype(int), 0, mask.shape[1]-1)
    #     x_coords = np.clip(surface_voxels[:, 2].astype(int), 0, mask.shape[2]-1)
    #     
    #     # Get gradients at surface points
    #     gx = grad_x[z_coords, y_coords, x_coords]
    #     gy = grad_y[z_coords, y_coords, x_coords]
    #     gz = grad_z[z_coords, y_coords, x_coords]
    #     
    #     # Stack and normalize
    #     normals = np.column_stack([gx, gy, gz])
    #     norms = np.linalg.norm(normals, axis=1, keepdims=True)
    #     norms[norms == 0] = 1.0  # Avoid division by zero
    #     normals = normals / norms
    #     
    #     return normals
    
    # def _fast_spatial_sampling(self, 
    #                          surface_voxels: np.ndarray, 
    #                          normals: np.ndarray,
    #                          min_distance: float,
    #                          target_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Fast spatial sampling using grid-based rejection.
    #     """
    #     if len(surface_voxels) == 0:
    #         return np.array([]), np.array([])
    #     
    #     # Create spatial hash grid for fast distance queries
    #     grid_size = max(1, int(min_distance))
    #     grid = {}
    #     
    #     def _get_grid_key(pos):
    #         return tuple((pos / grid_size).astype(int))
    #     
    #     def _is_valid_sample(pos, grid, min_dist):
    #         """Check if position is valid (not too close to existing samples)"""
    #         key = _get_grid_key(pos)
    #         # Check neighboring grid cells
    #         for dz in [-1, 0, 1]:
    #             for dy in [-1, 0, 1]:
    #                 for dx in [-1, 0, 1]:
    #                     neighbor_key = (key[0] + dz, key[1] + dy, key[2] + dx)
    #                     if neighbor_key in grid:
    #                         for existing_pos in grid[neighbor_key]:
    #                             if np.linalg.norm(pos - existing_pos) < min_dist:
    #                                 return False
    #         return True
    #     
    #     # Progressive sampling
    #     selected_centers = []
    #     selected_normals = []
    #     
    #     # Shuffle surface voxels for random sampling
    #     indices = np.random.permutation(len(surface_voxels))
    #     
    #     for idx in indices:
    #         if len(selected_centers) >= target_samples:
    #             break
    #             
    #         pos = surface_voxels[idx]
    #         
    #         if _is_valid_sample(pos, grid, min_distance):
    #             selected_centers.append(pos)
    #             selected_normals.append(normals[idx])
    #             
    #             # Add to spatial grid
    #             key = _get_grid_key(pos)
    #             if key not in grid:
    #                 grid[key] = []
    #             grid[key].append(pos)
    #     
    #     return np.array(selected_centers), np.array(selected_normals)
    
    def sample_mesh_faces_with_sdf_normals(self, 
                                         mesh: o3d.geometry.TriangleMesh,
                                         phi: np.ndarray,
                                         min_distance: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample mesh faces and compute normals aligned with SDF gradient direction.
        
        This method samples faces from the mesh and computes normals that are
        aligned with the SDF gradient direction (pointing towards positive phi values).
        
        Args:
            mesh: Open3D triangle mesh
            phi: Signed distance field
            min_distance: Minimum distance between sampled faces
            
        Returns:
            Tuple of (centers, normals) where:
            - centers: Face center coordinates (K, 3)
            - normals: Face normals aligned with SDF gradient (K, 3)
        """
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return np.array([]), np.array([])
        
        # Get mesh data
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # Compute face centers
        face_centers = vertices[faces].mean(axis=1)  # (N_faces, 3)
        
        # Compute face normals using cross product
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]] 
        v2 = vertices[faces[:, 2]]
        
        # Face normals using cross product (v1-v0) x (v2-v0)
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)
        
        # Normalize face normals
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        face_normals = face_normals / norms
        
        # Align normals with SDF gradient direction
        aligned_normals = self._align_normals_with_sdf_gradient(
            face_centers, face_normals, phi
        )
        
        # Sample faces with minimum distance constraint
        sampled_centers, sampled_normals = self._sample_faces_with_distance_constraint(
            face_centers, aligned_normals, min_distance
        )
        
        return sampled_centers, sampled_normals
    
    def _align_normals_with_sdf_gradient(self, 
                                       centers: np.ndarray, 
                                       normals: np.ndarray,
                                       phi: np.ndarray) -> np.ndarray:
        """
        Align face normals with SDF gradient direction.
        
        Args:
            centers: Face center coordinates (K, 3)
            normals: Face normals (K, 3)
            phi: Signed distance field
            
        Returns:
            Aligned normals pointing towards positive phi values
        """
        # Compute SDF gradients
        gz, gy, gx = np.gradient(phi.astype(np.float32))
        
        # Sample gradients at face centers
        z_coords = np.clip(np.rint(centers[:, 2]).astype(int), 0, phi.shape[0] - 1)
        y_coords = np.clip(np.rint(centers[:, 1]).astype(int), 0, phi.shape[1] - 1)
        x_coords = np.clip(np.rint(centers[:, 0]).astype(int), 0, phi.shape[2] - 1)
        
        # Get SDF gradients at face centers
        sdf_grad_x = gx[z_coords, y_coords, x_coords]
        sdf_grad_y = gy[z_coords, y_coords, x_coords]
        sdf_grad_z = gz[z_coords, y_coords, x_coords]
        
        # Stack SDF gradients
        sdf_gradients = np.column_stack([sdf_grad_x, sdf_grad_y, sdf_grad_z])
        
        # Normalize SDF gradients
        sdf_norms = np.linalg.norm(sdf_gradients, axis=1, keepdims=True)
        sdf_norms[sdf_norms == 0] = 1.0
        sdf_gradients = sdf_gradients / sdf_norms
        
        # Align normals with SDF gradient direction
        # If dot product is negative, flip the normal
        dots = np.sum(normals * sdf_gradients, axis=1)
        aligned_normals = normals.copy()
        aligned_normals[dots < 0] = -aligned_normals[dots < 0]
        
        return aligned_normals
    
    def _sample_faces_with_distance_constraint(self,
                                             face_centers: np.ndarray,
                                             face_normals: np.ndarray,
                                             min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample faces with minimum distance constraint using spatial hashing.
        
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
        Apply non-maximum suppression to remove overlapping samples.
        
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
                             volume_shape: Optional[Tuple[int, int, int]] = None,
                             use_fast_sampling: bool = False,  # Disabled by default
                             target_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to sample points on membrane surface using mesh face sampling.
        
        This method now uses mesh face sampling with SDF-aligned normals instead
        of the deprecated fast voxel-based sampling.
        
        Args:
            mask: Binary membrane mask (0/1)
            particle_radius: Effective particle radius for boundary checking
            volume_shape: Volume dimensions for boundary checking
            use_fast_sampling: DEPRECATED - no longer used
            target_samples: DEPRECATED - no longer used
            
        Returns:
            Tuple of (centers, normals) where:
            - centers: Sampled point coordinates (K, 3)
            - normals: Corresponding surface normals aligned with SDF gradient (K, 3)
        """
        if volume_shape is None:
            volume_shape = mask.shape[::-1]  # Convert from (z,y,x) to (x,y,z)
        
        print("Using mesh face sampling with SDF-aligned normals...")
        
        # Step 1: Create signed distance field
        phi = self.create_signed_distance_field(mask)
        
        # Step 2: Extract mesh from SDF
        mesh = self.extract_mesh_from_sdf(phi)
        
        # Step 3: Sample mesh faces with SDF-aligned normals
        centers, normals = self.sample_mesh_faces_with_sdf_normals(
            mesh, phi, self.min_distance
        )
        
        # Step 4: Apply non-maximum suppression
        centers, normals = self.apply_non_maximum_suppression(
            centers, normals, self.min_distance
        )
        
        # Step 5: Check placement feasibility
        centers, normals = self.check_placement_feasibility(
            centers, normals, volume_shape, particle_radius
        )
        
        return centers, normals

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

    @staticmethod
    def _compute_sdf_gradient(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients of SDF in z, y, x order (matching phi layout)."""
        gz, gy, gx = np.gradient(phi.astype(np.float32))
        return gz, gy, gx

    def _orient_mesh_normals_outward(self, mesh: o3d.geometry.TriangleMesh, phi: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Orient mesh normals to point outward, defined as the direction of increasing SDF.

        Strategy:
        - Compute SDF gradients on the voxel grid.
        - For each vertex, sample the nearest gradient (convert vertex x,y,z -> z,y,x index order).
        - If the average dot between vertex normals and sampled gradients is negative,
          flip triangle winding to invert normals and recompute.
        """
        if len(mesh.vertices) == 0:
            return mesh

        gz, gy, gx = self._compute_sdf_gradient(phi)
        verts = np.asarray(mesh.vertices)

        # Map vertex positions to nearest voxel indices (z,y,x). verts are in (x,y,z)
        vz = np.clip(np.rint(verts[:, 2]).astype(int), 0, phi.shape[0] - 1)
        vy = np.clip(np.rint(verts[:, 1]).astype(int), 0, phi.shape[1] - 1)
        vx = np.clip(np.rint(verts[:, 0]).astype(int), 0, phi.shape[2] - 1)

        grad = np.stack([
            gx[vz, vy, vx],  # x component
            gy[vz, vy, vx],  # y component
            gz[vz, vy, vx],  # z component
        ], axis=1)

        # Normalize gradients; avoid division by zero
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm == 0] = 1.0
        grad_unit = grad / grad_norm

        # Ensure vertex normals are available
        mesh.compute_vertex_normals()
        vnormals = np.asarray(mesh.vertex_normals)

        # Compute mean alignment
        dots = np.einsum('ij,ij->i', vnormals, grad_unit)
        mean_dot = float(np.nanmean(dots))

        if mean_dot < 0.0:
            # Flip triangle winding to invert normals globally
            tris = np.asarray(mesh.triangles)
            tris_flipped = tris[:, ::-1]
            mesh.triangles = o3d.utility.Vector3iVector(tris_flipped)
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()

        return mesh

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


def create_mesh_geodesic_sampler(min_distance: float = 20.0,
                                smoothing_sigma: float = 1.5,
                                taubin_iterations: int = 10) -> MeshGeodesicSampler:
    """
    Factory function to create a MeshGeodesicSampler instance.
    
    Args:
        min_distance: Minimum distance between sampling points (in voxels)
        smoothing_sigma: Gaussian smoothing parameter (in voxels)
        taubin_iterations: Number of Taubin smoothing iterations
        
    Returns:
        Configured MeshGeodesicSampler instance
    """
    return MeshGeodesicSampler(
        smoothing_sigma=smoothing_sigma,
        taubin_iterations=taubin_iterations,
        min_distance=min_distance
    )


# ---------------------------
# High-level helper utilities
# ---------------------------

def generate_synthetic_mask(
    shape: Tuple[int, int, int] = (100, 100, 100),
    center: Tuple[int, int, int] = (50, 50, 50),
    radius: int = 30,
) -> np.ndarray:
    """
    Create a simple spherical binary mask for testing.
    
    Args:
        shape: Volume shape (z, y, x)
        center: Sphere center (z, y, x)
        radius: Sphere radius in voxels
    
    Returns:
        Binary uint8 mask with values in {0, 1}
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    mask = (distance <= radius).astype(np.uint8)
    return mask


def run_mesh_geodesic_sampling(
    mask: np.ndarray,
    *, # '*' is used to indicate that the following parameters are keyword-only
    min_distance: float = 20.0,
    smoothing_sigma: float = 1.5,
    taubin_iterations: int = 10,
    particle_radius: float = 10.0,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    use_fast_sampling: bool = False,  # DEPRECATED - no longer used
    target_samples: int = 100000,    # DEPRECATED - no longer used
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-shot convenience wrapper to run mesh-geodesic sampling on a mask.
    
    Now uses mesh face sampling with SDF-aligned normals for all datasets.
    
    Args:
        mask: Binary membrane mask (0/1)
        min_distance: Minimum distance between samples (in voxels)
        smoothing_sigma: Gaussian smoothing parameter (in voxels)
        taubin_iterations: Number of Taubin smoothing iterations
        particle_radius: Effective particle radius for boundary checking
        volume_shape: Volume dimensions for boundary checking
        use_fast_sampling: DEPRECATED - no longer used
        target_samples: DEPRECATED - no longer used
    
    Returns:
        Tuple of (centers, normals) where:
        - centers: Sampled point coordinates (K, 3)
        - normals: Corresponding surface normals aligned with SDF gradient (K, 3)
    """
    sampler = create_mesh_geodesic_sampler(
        min_distance=min_distance,
        smoothing_sigma=smoothing_sigma,
        taubin_iterations=taubin_iterations,
    )
    centers, normals = sampler.sample_membrane_points(
        mask,
        particle_radius=particle_radius,
        volume_shape=volume_shape,
        use_fast_sampling=use_fast_sampling,
        target_samples=target_samples,
    )
    return centers, normals


def save_sampling_outputs(
    output_dir: Union[str, Path],
    centers: np.ndarray,
    normals: np.ndarray,
    *,
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    sigma_tilt: float = 30.0,
    sigma_psi: float = 30.0,
    sigma_rot: float = 30.0,
    create_vis_script: bool = False,
    use_subtomogram_format: bool = True,
) -> Tuple[Path, Path, Path, Optional[Path]]:
    """
    Save standard outputs produced by sampling.
    
    Files:
      - sampling_coordinates.csv (centers + normals)
      - particles.star (RELION 5 subtomogram STAR format)
      - coordinates.csv (x,y,z + tilt,psi,rot)
      - prior_angles.csv (RELION priors)
      - visualize_results.py (optional viewer script)
    
    Args:
        use_subtomogram_format: If True, use RELION 5 subtomogram format
                                with proper header and column order
    
    Returns tuple of generated file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Coordinates CSV with normals
    coord_csv = out_dir / "sampling_coordinates.csv"
    df_coords = pd.DataFrame(
        np.column_stack([centers, normals]),
        columns=["x", "y", "z", "nx", "ny", "nz"],
    )
    df_coords.to_csv(coord_csv, index=False)

    # 2) RELION STAR with subtomogram format
    star_file = out_dir / "particles.star"
    if use_subtomogram_format:
        # Use RELION 5 subtomogram format
        from tomopanda.utils.mrc_utils import save_subtomogram_coordinates
        save_subtomogram_coordinates(
            centers,
            normals,
            star_file,
            tomogram_name=tomogram_name,
            particle_diameter=particle_diameter,
            voxel_size=voxel_size,
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
        voxel_size=voxel_size,
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

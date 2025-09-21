"""
Mesh Geodesic Sampling for CryoET Particle Picking

This module implements mesh-geodesic sampling algorithm for membrane-based
particle picking in cryoET. The algorithm converts membrane masks to triangular
meshes and performs geodesic-based sampling to generate particle picking candidates.

Author: TomoPANDA Team
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import open3d as o3d


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
        
        return mesh
    
    def geodesic_farthest_point_sampling(self, 
                                       vertices: np.ndarray, 
                                       faces: np.ndarray,
                                       radius: float) -> np.ndarray:
        """
        Perform geodesic farthest point sampling on mesh.
        
        This approximates Poisson-disk sampling using geodesic distances.
        
        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            radius: Minimum geodesic distance between samples
            
        Returns:
            Array of vertex indices for sampled points
        """
        try:
            import gdist
        except ImportError:
            raise ImportError("gdist library is required for geodesic sampling. "
                            "Install with: pip install gdist")
        
        picked = []
        n_vertices = len(vertices)
        
        # Start with vertex closest to centroid
        centroid = vertices.mean(0)
        start_idx = np.argmin(np.linalg.norm(vertices - centroid, axis=1))
        picked.append(start_idx)
        
        # Maintain minimum geodesic distance to picked set
        dmin = np.full(n_vertices, np.inf)
        
        for _ in range(100000):  # Upper limit, will break early
            # Compute geodesic distances from latest seed point
            d = gdist.compute_gdist(
                vertices.astype(np.float64), 
                faces, 
                np.array([picked[-1]], dtype=np.int32)
            )
            dmin = np.minimum(dmin, d)
            
            # Select vertex with maximum minimum distance
            next_idx = int(np.argmax(dmin))
            
            if dmin[next_idx] < radius:
                # All vertices are within radius of picked set
                break
                
            picked.append(next_idx)
            
        return np.array(picked, dtype=int)
    
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
                             volume_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to sample points on membrane surface.
        
        Args:
            mask: Binary membrane mask (0/1)
            particle_radius: Effective particle radius for boundary checking
            volume_shape: Volume dimensions for boundary checking
            
        Returns:
            Tuple of (centers, normals) where:
            - centers: Sampled point coordinates (K, 3)
            - normals: Corresponding surface normals (K, 3)
        """
        if volume_shape is None:
            volume_shape = mask.shape[::-1]  # Convert from (z,y,x) to (x,y,z)
        
        # Step 1: Create signed distance field
        phi = self.create_signed_distance_field(mask)
        
        # Step 2: Extract mesh from SDF
        mesh = self.extract_mesh_from_sdf(phi)
        
        # Step 3: Get mesh data
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        
        # Step 4: Geodesic sampling
        sampled_indices = self.geodesic_farthest_point_sampling(
            vertices, faces, self.min_distance
        )
        
        # Step 5: Extract centers and normals
        centers = vertices[sampled_indices]
        normals = np.asarray(mesh.vertex_normals)[sampled_indices]
        
        # Step 6: Apply non-maximum suppression
        centers, normals = self.apply_non_maximum_suppression(
            centers, normals, self.min_distance
        )
        
        # Step 7: Check placement feasibility
        centers, normals = self.check_placement_feasibility(
            centers, normals, volume_shape, particle_radius
        )
        
        return centers, normals


def create_mesh_geodesic_sampler(min_distance: float = 20.0,
                                smoothing_sigma: float = 1.5,
                                taubin_iterations: int = 10) -> MeshGeodesicSampler:
    """
    Factory function to create a MeshGeodesicSampler instance.
    
    Args:
        min_distance: Minimum distance between sampling points
        smoothing_sigma: Gaussian smoothing parameter
        taubin_iterations: Number of Taubin smoothing iterations
        
    Returns:
        Configured MeshGeodesicSampler instance
    """
    return MeshGeodesicSampler(
        smoothing_sigma=smoothing_sigma,
        taubin_iterations=taubin_iterations,
        min_distance=min_distance
    )

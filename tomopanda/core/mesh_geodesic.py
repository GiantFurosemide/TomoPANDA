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
        # Build mesh edge graph once (used if gdist is unavailable)
        def _build_mesh_graph(verts: np.ndarray, tris: np.ndarray) -> csr_matrix:
            i_list = []
            j_list = []
            w_list = []
            # Each triangle contributes three undirected edges
            for f in tris:
                a, b, c = int(f[0]), int(f[1]), int(f[2])
                for u, v in ((a, b), (b, c), (c, a)):
                    if u == v:
                        continue
                    duv = np.linalg.norm(verts[u] - verts[v])
                    i_list.extend([u, v])
                    j_list.extend([v, u])
                    w_list.extend([duv, duv])
            n = len(verts)
            graph = coo_matrix((w_list, (i_list, j_list)), shape=(n, n))
            return graph.tocsr()

        # Try to use gdist if available (fast and accurate). Otherwise fall back to graph shortest paths.
        try:
            import gdist  # type: ignore
            use_gdist = True
        except Exception:
            use_gdist = False
            graph_csr = _build_mesh_graph(vertices, faces)
        
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
            if use_gdist:
                d = gdist.compute_gdist(
                    vertices.astype(np.float64), 
                    faces, 
                    np.array([picked[-1]], dtype=np.int32)
                )
            else:
                # Single-source shortest paths on the edge-weighted mesh graph
                d = dijkstra(graph_csr, directed=False, indices=picked[-1])
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
        
        # Step 0: Optional: identify surface voxels and outward side for reference
        surface_mask = self.extract_surface_voxels(mask)

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
    *,
    min_distance: float = 20.0,
    smoothing_sigma: float = 1.5,
    taubin_iterations: int = 10,
    particle_radius: float = 10.0,
    volume_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-shot convenience wrapper to run mesh-geodesic sampling on a mask.
    
    Returns centers and normals.
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
) -> Tuple[Path, Path, Path, Optional[Path]]:
    """
    Save standard outputs produced by sampling.
    
    Files:
      - sampling_coordinates.csv (centers + normals)
      - particles.star (RELION STAR)
      - coordinates.csv (x,y,z + tilt,psi,rot)
      - prior_angles.csv (RELION priors)
      - visualize_results.py (optional viewer script)
    
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

    # 2) RELION STAR
    star_file = out_dir / "particles.star"
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

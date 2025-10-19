"""
Voxel-based surface extraction and sampling for CryoET masks.

Implements two public functions following `tomopanda/doc/voxel_sample.md`:

- info_extract(mask):
  Inputs a binary mask volume (X, Y, Z) with values in {0, 1} and returns:
    1) surface_mask: ndarray shape (X, Y, Z), dtype uint8, where 1 marks mask
       voxels that are adjacent to at least one background voxel under
       6-connectivity.
    2) orientations: ndarray shape (3, X, Y, Z), dtype float32, where each
       vector at a surface voxel is the average of unit vectors pointing from
       the center of the mask voxel to the centers of all adjacent background
       neighbor voxels (6-neighborhood). If none, vector is (0, 0, 0).

- sample(min_distance, edge_distance, surface_mask, orientations):
  Returns a dense field ndarray of shape (6, X, Y, Z) that stores per-voxel
  candidate entries (x, y, z, vx, vy, vz) for selected surface voxels under a
  minimum Euclidean distance constraint between selected centers. Unselected
  voxels remain zeros.

Coordinate convention: inputs and outputs are in (X, Y, Z).
"""

from typing import Tuple
from typing import Optional, Union
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D (X,Y,Z), got shape {mask.shape}")
    # Force 0/1
    mask_bin = (mask > 0).astype(np.uint8)
    return mask_bin


def _six_connectivity_structure() -> np.ndarray:
    # 6-neighborhood structuring element in (X,Y,Z) indexing
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, 1] = True
    st[0, 1, 1] = True  # -X
    st[2, 1, 1] = True  # +X
    st[1, 0, 1] = True  # -Y
    st[1, 2, 1] = True  # +Y
    st[1, 1, 0] = True  # -Z
    st[1, 1, 2] = True  # +Z
    return st


def info_extract(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract surface voxels and per-voxel orientation field.

    Args:
        mask: ndarray (X, Y, Z) with values {0,1}

    Returns:
        surface_mask: ndarray (X, Y, Z), dtype uint8
        orientations: ndarray (3, X, Y, Z), dtype float32
    """
    mask_bin = _validate_mask(mask)

    # Surface detection via 6-connected erosion difference
    structure = _six_connectivity_structure()
    eroded = binary_erosion(mask_bin.astype(bool), structure=structure)
    surface = mask_bin.astype(bool) & (~eroded)
    surface_mask = surface.astype(np.uint8)

    # Orientation computation: average of unit vectors from mask voxel to adjacent background voxels
    sx, sy, sz = mask_bin.shape
    orientations = np.zeros((3, sx, sy, sz), dtype=np.float32)

    # Neighbor offsets for 6-connectivity in (X,Y,Z)
    neighbor_offsets = np.array([
        [-1, 0, 0],  # -X
        [1, 0, 0],   # +X
        [0, -1, 0],  # -Y
        [0, 1, 0],   # +Y
        [0, 0, -1],  # -Z
        [0, 0, 1],   # +Z
    ], dtype=int)

    surface_indices = np.stack(np.nonzero(surface_mask), axis=1)  # (N, 3) in (X,Y,Z)

    for idx in surface_indices:
        x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
        vec_sum = np.zeros(3, dtype=np.float32)
        count = 0
        for off in neighbor_offsets:
            xn, yn, zn = x + int(off[0]), y + int(off[1]), z + int(off[2])
            if 0 <= xn < sx and 0 <= yn < sy and 0 <= zn < sz:
                if mask_bin[xn, yn, zn] == 0:  # background neighbor
                    # Vector from center (x,y,z) to neighbor center (xn,yn,zn)
                    direction = np.array([float(xn - x), float(yn - y), float(zn - z)], dtype=np.float32)
                    # Under 6-neighborhood this is axis-aligned with unit length=1
                    # Normalize to be safe
                    norm = float(np.linalg.norm(direction))
                    if norm > 0.0:
                        vec_sum += direction / norm
                        count += 1
        if count > 0:
            # Average and normalize to unit vector
            vec = vec_sum / float(count)
            n = float(np.linalg.norm(vec))
            if n > 0.0:
                vec = vec / n
            orientations[:, x, y, z] = vec.astype(np.float32)
        else:
            orientations[:, x, y, z] = 0.0

    return surface_mask, orientations


def sample(
    min_distance: float,
    edge_distance: float,
    surface_mask: np.ndarray,
    orientations: np.ndarray,
) -> np.ndarray:
    """
    Sample surface voxels using min_distance for spacing and edge_distance as edge margin.

    Args:
        min_distance: minimum Euclidean distance between selected centers (in pixels)
        edge_distance: edge distance margin (in pixels); do not sample within this
                       distance from any box boundary to avoid half particles
        surface_mask: ndarray (X, Y, Z) from info_extract
        orientations: ndarray (3, X, Y, Z) from info_extract

    Returns:
        field: ndarray (6, X, Y, Z) with (x, y, z, vx, vy, vz) at selected voxels, zeros elsewhere
    """
    if surface_mask.ndim != 3:
        raise ValueError("surface_mask must be (X,Y,Z)")
    if orientations.shape != (3,) + surface_mask.shape:
        raise ValueError("orientations must be (3,X,Y,Z) aligned with surface_mask")

    sx, sy, sz = surface_mask.shape
    field = np.zeros((6, sx, sy, sz), dtype=np.float32)

    # Collect candidate positions
    candidates = np.stack(np.nonzero(surface_mask > 0), axis=1).astype(np.int32)  # (N,3)
    # Exclude candidates too close to the volume boundary to avoid half particles
    margin = int(np.ceil(float(max(0.0, edge_distance))))
    if margin > 0:
        if sx - 1 - margin < margin or sy - 1 - margin < margin or sz - 1 - margin < margin:
            # Volume too small for the requested margin; return empty field
            return field
        inside = (
            (candidates[:, 0] >= margin)
            & (candidates[:, 0] <= sx - 1 - margin)
            & (candidates[:, 1] >= margin)
            & (candidates[:, 1] <= sy - 1 - margin)
            & (candidates[:, 2] >= margin)
            & (candidates[:, 2] <= sz - 1 - margin)
        )
        candidates = candidates[inside]
    if candidates.size == 0:
        return field

    # Shuffle for random selection
    rng = np.random.default_rng()
    order = rng.permutation(candidates.shape[0])
    candidates = candidates[order]

    # Simple grid-based rejection sampling using a spatial hash
    spacing = float(min_distance)
    if spacing <= 0.0:
        spacing = 1.0
    grid_size = max(1, int(round(spacing)))
    grid = {}

    def grid_key(pos: np.ndarray) -> Tuple[int, int, int]:
        return (int(pos[0] // grid_size), int(pos[1] // grid_size), int(pos[2] // grid_size))

    def is_far_enough(pos: np.ndarray) -> bool:
        k = grid_key(pos)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = (k[0] + dx, k[1] + dy, k[2] + dz)
                    if nk in grid:
                        for q in grid[nk]:
                            if np.linalg.norm(pos.astype(np.float32) - q.astype(np.float32)) < spacing:
                                return False
        return True

    for x, y, z in candidates:
        p = np.array([x, y, z], dtype=np.float32)
        if not is_far_enough(p):
            continue
        # Accept
        k = grid_key(p)
        grid.setdefault(k, []).append(p)

        vx, vy, vz = orientations[:, x, y, z]
        field[0, x, y, z] = float(x)
        field[1, x, y, z] = float(y)
        field[2, x, y, z] = float(z)
        field[3, x, y, z] = float(vx)
        field[4, x, y, z] = float(vy)
        field[5, x, y, z] = float(vz)

    return field


def extract_centers_and_normals_from_field(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sampling centers and normals from a dense field produced by sample().

    Args:
        field: ndarray (6, X, Y, Z) storing (x, y, z, vx, vy, vz) at selected voxels

    Returns:
        centers: ndarray (K, 3) with xyz coordinates
        normals: ndarray (K, 3) with unit normal vectors
    """
    if field.ndim != 4 or field.shape[0] != 6:
        raise ValueError("field must be (6, X, Y, Z)")

    xch, ych, zch = field[0], field[1], field[2]
    vxch, vych, vzch = field[3], field[4], field[5]

    # Consider a voxel selected if any of (x,y,z) channels are non-zero
    selected = (xch != 0) | (ych != 0) | (zch != 0)
    idx = np.stack(np.nonzero(selected), axis=1)  # (K, 3) indices in (X,Y,Z)
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Gather values
    centers = np.stack([
        xch[selected].astype(np.float32),
        ych[selected].astype(np.float32),
        zch[selected].astype(np.float32),
    ], axis=1)

    normals = np.stack([
        vxch[selected].astype(np.float32),
        vych[selected].astype(np.float32),
        vzch[selected].astype(np.float32),
    ], axis=1)

    # Normalize normals defensively
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    n[n == 0] = 1.0
    normals = normals / n

    return centers, normals


def save_field_as_relion_star(
    field: np.ndarray,
    output_path: Union[str, Path],
    *,
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Save the voxel sampling field as a RELION 5 particle STAR file.

    Args:
        field: ndarray (6, X, Y, Z) produced by sample()
        output_path: destination .star path
        tomogram_name: RELION tomogram name
        particle_diameter: particle diameter in Angstroms
        voxel_size: optional (sx, sy, sz) Angstrom scaling applied to centers
    """
    centers, normals = extract_centers_and_normals_from_field(field)

    # Optional scaling by voxel size for physical units
    if voxel_size is not None and len(centers) > 0:
        centers = centers * np.array(voxel_size, dtype=np.float32)

    # Use simplified RELION format (like mesh_geodesic.py)
    from tomopanda.core.mesh_geodesic import _save_simplified_relion_star

    output_path = Path(output_path)
    _save_simplified_relion_star(
        centers=centers,
        normals=normals,
        output_path=output_path,
        tomogram_name=tomogram_name,
        particle_diameter=particle_diameter,
        voxel_size=voxel_size,
    )


def run_voxel_sampling_to_star(
    mask: np.ndarray,
    *,
    min_distance: float,
    edge_distance: float,
    output_path: Union[str, Path],
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: run info_extract + sample, then write RELION STAR.

    Returns centers and normals for further use.
    """
    surface_mask, orientations = info_extract(mask)
    field = sample(
        min_distance=min_distance,
        edge_distance=edge_distance,
        surface_mask=surface_mask,
        orientations=orientations,
    )
    save_field_as_relion_star(
        field,
        output_path,
        tomogram_name=tomogram_name,
        particle_diameter=particle_diameter,
        voxel_size=voxel_size,
    )
    centers, normals = extract_centers_and_normals_from_field(field)
    return centers, normals


__all__ = [
    "info_extract",
    "sample",
    "extract_centers_and_normals_from_field",
    "save_field_as_relion_star",
    "run_voxel_sampling_to_star",
]



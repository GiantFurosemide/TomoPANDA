"""
RELION Format Utilities for TomoPANDA

This module provides utilities for converting mesh geodesic sampling results
to RELION format for particle picking and 3D classification.

Author: TomoPANDA Team
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict
from pathlib import Path
import math


class RELIONConverter:
    """
    Utility class for converting coordinates and orientations to RELION format.
    """
    
    @staticmethod
    def normal_to_euler(normal: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert surface normal to Euler angles (tilt, psi, rot).
        
        This method converts a surface normal vector to RELION ZYZ Euler angles
        such that the z-axis (0,0,1) rotated by these angles points in the
        direction of the normal vector.
        
        Args:
            normal: Surface normal vector (3,) - should be normalized
            
        Returns:
            Tuple of (tilt, psi, rot) in degrees
        """
        # Normalize the input vector
        normal = normal / np.linalg.norm(normal)
        nx, ny, nz = normal
        
        # Calculate tilt (angle from z-axis)
        # tilt = 0 when normal points along +z, tilt = 180 when along -z
        tilt = math.degrees(math.acos(np.clip(nz, -1.0, 1.0)))
        
        # Calculate psi (rotation around z-axis in xy-plane)
        # This determines the direction in the xy-plane
        psi = math.degrees(math.atan2(ny, nx))
        
        # For membrane normals, we typically don't need in-plane rotation
        # The rot angle represents rotation around the normal itself
        rot = 0.0
        
        return tilt, psi, rot
    
    @staticmethod
    def euler_to_rotation_matrix(tilt: float, psi: float, rot: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix using RELION ZYZ convention.
        
        Args:
            tilt: Tilt angle in degrees
            psi: Psi angle in degrees  
            rot: Rot angle in degrees
            
        Returns:
            3x3 rotation matrix
        """
        # Convert to radians
        tilt_rad = math.radians(tilt)
        psi_rad = math.radians(psi)
        rot_rad = math.radians(rot)
        
        # RELION ZYZ Euler angle convention
        # R = Rz(rot) * Ry(tilt) * Rz(psi)
        
        # First rotation: psi around Z-axis
        Rz_psi = np.array([
            [math.cos(psi_rad), -math.sin(psi_rad), 0],
            [math.sin(psi_rad), math.cos(psi_rad), 0],
            [0, 0, 1]
        ])
        
        # Second rotation: tilt around Y-axis
        Ry_tilt = np.array([
            [math.cos(tilt_rad), 0, math.sin(tilt_rad)],
            [0, 1, 0],
            [-math.sin(tilt_rad), 0, math.cos(tilt_rad)]
        ])
        
        # Third rotation: rot around Z-axis
        Rz_rot = np.array([
            [math.cos(rot_rad), -math.sin(rot_rad), 0],
            [math.sin(rot_rad), math.cos(rot_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz(rot) * Ry(tilt) * Rz(psi)
        R = Rz_rot @ Ry_tilt @ Rz_psi
        
        return R

    @staticmethod
    def rotation_matrix_to_direction(tilt: float, psi: float, rot: float, axis: str = 'z') -> np.ndarray:
        """
        Convert Euler angles to a unit direction vector by rotating a basis axis.

        Args:
            tilt: Tilt angle in degrees
            psi: Psi angle in degrees
            rot: Rot angle in degrees
            axis: One of 'x', 'y', 'z' indicating which basis axis to rotate

        Returns:
            Unit 3D direction vector (3,)
        """
        R = RELIONConverter.euler_to_rotation_matrix(tilt, psi, rot)
        if axis == 'x':
            v = np.array([1.0, 0.0, 0.0], dtype=float)
        elif axis == 'y':
            v = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            v = np.array([0.0, 0.0, 1.0], dtype=float)
        d = R @ v
        n = np.linalg.norm(d)
        return d / (n if n > 0 else 1.0)
    
    @staticmethod
    def create_star_file(centers: np.ndarray, 
                        normals: np.ndarray,
                        output_path: Union[str, Path],
                        tomogram_name: str = "tomogram",
                        particle_diameter: float = 200.0,
                        confidence: float = 0.5) -> None:
        """
        Create RELION STAR file for subtomogram particle coordinates.
        
        Uses RELION 5 subtomogram tags: rlnTomoSubtomogramRot, rlnTomoSubtomogramTilt, rlnTomoSubtomogramPsi
        
        Args:
            centers: Sample centers (K, 3)
            normals: Sample normals (K, 3) - membrane surface normals
            output_path: Output STAR file path
            tomogram_name: Name of the tomogram
            particle_diameter: Particle diameter in Angstroms
            confidence: Confidence score for particles
        """
        if len(centers) == 0:
            raise ValueError("No coordinates to save")
        
        # Convert membrane normals to Euler angles for subtomogram orientation
        euler_angles = []
        for normal in normals:
            tilt, psi, rot = RELIONConverter.normal_to_euler(normal)
            euler_angles.append([tilt, psi, rot])
        
        euler_angles = np.array(euler_angles)
        
        # Create STAR file data with RELION 5 subtomogram tags
        data = {
            'rlnCoordinateX': centers[:, 0],
            'rlnCoordinateY': centers[:, 1], 
            'rlnCoordinateZ': centers[:, 2],
            # RELION 5 subtomogram rotation tags
            'rlnTomoSubtomogramRot': euler_angles[:, 2],    # rot angle
            'rlnTomoSubtomogramTilt': euler_angles[:, 0],   # tilt angle  
            'rlnTomoSubtomogramPsi': euler_angles[:, 1],    # psi angle
            # Legacy angle tags for compatibility
            'rlnAngleTilt': euler_angles[:, 0],
            'rlnAnglePsi': euler_angles[:, 1],
            'rlnAngleRot': euler_angles[:, 2],
            'rlnTomoName': [tomogram_name] * len(centers),
            'rlnTomoParticleId': range(len(centers)),
            'rlnClassNumber': [1] * len(centers),
            'rlnAutopickFigureOfMerit': [confidence] * len(centers),
            'rlnCtfMaxResolution': [4.0] * len(centers),
            'rlnCtfFigureOfMerit': [0.8] * len(centers),
            'rlnCtfBfactor': [0.0] * len(centers),
            'rlnCtfScalefactor': [1.0] * len(centers),
            'rlnDefocusU': [0.0] * len(centers),
            'rlnDefocusV': [0.0] * len(centers),
            'rlnDefocusAngle': [0.0] * len(centers),
            'rlnPhaseShift': [0.0] * len(centers),
            'rlnCtfValue': [0.0] * len(centers),
            'rlnGroupNumber': [1] * len(centers),
            'rlnOriginX': [0.0] * len(centers),
            'rlnOriginY': [0.0] * len(centers),
            'rlnOriginZ': [0.0] * len(centers),
            'rlnRandomSubset': [1] * len(centers),
            'rlnParticleSelectZScore': [0.0] * len(centers),
            'rlnHelicalTubeID': [0] * len(centers),
            'rlnHelicalTrackLength': [0.0] * len(centers),
            # Prior angles for subtomogram averaging
            'rlnAngleTiltPrior': euler_angles[:, 0],
            'rlnAnglePsiPrior': euler_angles[:, 1],
            'rlnAngleRotPrior': euler_angles[:, 2],
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
        
        print(f"Saved {len(centers)} particles to RELION STAR file: {output_path}")
    
    @staticmethod
    def create_coordinate_file(centers: np.ndarray,
                             normals: np.ndarray,
                             output_path: Union[str, Path],
                             voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Create simple coordinate file with positions and orientations.
        
        Args:
            centers: Sample centers (K, 3)
            normals: Sample normals (K, 3)
            output_path: Output file path
            voxel_size: Voxel size for coordinate scaling
        """
        if len(centers) == 0:
            raise ValueError("No coordinates to save")
        
        # Scale coordinates if voxel size is provided
        if voxel_size is not None:
            centers_scaled = centers * np.array(voxel_size)
        else:
            centers_scaled = centers
        
        # Convert normals to Euler angles
        euler_angles = []
        for normal in normals:
            tilt, psi, rot = RELIONConverter.normal_to_euler(normal)
            euler_angles.append([tilt, psi, rot])
        
        euler_angles = np.array(euler_angles)
        
        # Create data array
        data = np.column_stack([
            centers_scaled,  # x, y, z coordinates
            euler_angles     # tilt, psi, rot angles
        ])
        
        # Save as CSV
        columns = ['x', 'y', 'z', 'tilt', 'psi', 'rot']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(centers)} coordinates to {output_path}")
    
    @staticmethod
    def create_prior_angles_file(centers: np.ndarray,
                               normals: np.ndarray,
                               output_path: Union[str, Path],
                               sigma_tilt: float = 30.0,
                               sigma_psi: float = 30.0,
                               sigma_rot: float = 30.0) -> None:
        """
        Create RELION prior angles file for 3D classification.
        
        Args:
            centers: Sample centers (K, 3)
            normals: Sample normals (K, 3)
            output_path: Output file path
            sigma_tilt: Standard deviation for tilt angle prior
            sigma_psi: Standard deviation for psi angle prior
            sigma_rot: Standard deviation for rot angle prior
        """
        if len(centers) == 0:
            raise ValueError("No coordinates to save")
        
        # Convert normals to Euler angles
        euler_angles = []
        for normal in normals:
            tilt, psi, rot = RELIONConverter.normal_to_euler(normal)
            euler_angles.append([tilt, psi, rot])
        
        euler_angles = np.array(euler_angles)
        
        # Create prior angles data
        data = {
            'rlnCoordinateX': centers[:, 0],
            'rlnCoordinateY': centers[:, 1],
            'rlnCoordinateZ': centers[:, 2],
            'rlnAngleTiltPrior': euler_angles[:, 0],
            'rlnAnglePsiPrior': euler_angles[:, 1],
            'rlnAngleRotPrior': euler_angles[:, 2],
            'rlnAngleTiltPriorSigma': [sigma_tilt] * len(centers),
            'rlnAnglePsiPriorSigma': [sigma_psi] * len(centers),
            'rlnAngleRotPriorSigma': [sigma_rot] * len(centers)
        }
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(centers)} prior angles to {output_path}")


def convert_to_relion_star(centers: np.ndarray,
                          normals: np.ndarray,
                          output_path: Union[str, Path],
                          tomogram_name: str = "tomogram",
                          particle_diameter: float = 200.0) -> None:
    """
    Convenience function to convert coordinates to RELION STAR format.
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3)
        output_path: Output STAR file path
        tomogram_name: Name of the tomogram
        particle_diameter: Particle diameter in Angstroms
    """
    RELIONConverter.create_star_file(
        centers, normals, output_path, tomogram_name, particle_diameter
    )


def convert_to_coordinate_file(centers: np.ndarray,
                              normals: np.ndarray,
                              output_path: Union[str, Path],
                              voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
    """
    Convenience function to create coordinate file.
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3)
        output_path: Output file path
        voxel_size: Voxel size for coordinate scaling
    """
    RELIONConverter.create_coordinate_file(
        centers, normals, output_path, voxel_size
    )


def convert_to_prior_angles(centers: np.ndarray,
                           normals: np.ndarray,
                           output_path: Union[str, Path],
                           sigma_tilt: float = 30.0,
                           sigma_psi: float = 30.0,
                           sigma_rot: float = 30.0) -> None:
    """
    Convenience function to create prior angles file.
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3)
        output_path: Output file path
        sigma_tilt: Standard deviation for tilt angle prior
        sigma_psi: Standard deviation for psi angle prior
        sigma_rot: Standard deviation for rot angle prior
    """
    RELIONConverter.create_prior_angles_file(
        centers, normals, output_path, sigma_tilt, sigma_psi, sigma_rot
    )


def read_star(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read a RELION .star file into a pandas DataFrame.

    Notes:
        - Supports simple 'data_ ... loop_' blocks with one table.
        - Lines starting with '#' or empty are ignored.
        - Column labels are expected as '_rln...' per RELION convention.

    Returns:
        DataFrame with columns stripped of leading underscore.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"STAR file not found: {filepath}")

    columns: List[str] = []
    rows: List[List[str]] = []
    in_loop = False

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('data_'):
                # start of a data block; continue
                continue
            if line.lower().startswith('loop_'):
                in_loop = True
                columns = []
                rows = []
                continue
            if in_loop and line.startswith('_'):
                # column label
                # keep original but drop leading underscore for DataFrame name
                # RELION may append numbering like "#17"; keep only the tag token
                # Example input: "_rlnCenteredCoordinateXAngst #17"
                # We want: "rlnCenteredCoordinateXAngst"
                col = line[1:].split()[0]
                columns.append(col)
                continue
            if in_loop:
                # data row (space-separated; RELION uses plain whitespace)
                parts = line.split()
                # allow rows shorter than columns by padding
                if len(parts) < len(columns):
                    parts += [''] * (len(columns) - len(parts))
                rows.append(parts[:len(columns)])

    if not columns:
        raise ValueError(f"No columns found in STAR file: {filepath}")

    df = pd.DataFrame(rows, columns=columns)

    # Try converting numeric columns to numbers
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')

    return df


def parse_particles_from_star(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract particle coordinates and Euler angles from a DataFrame created by read_star.
    
    Supports RELION 5 subtomogram format with proper coordinate and angle extraction.

    Returns dict with keys:
        - centers: (N,3) float array for particle coordinates
        - eulers: (N,3) float array for [tilt, psi, rot]
    """
    # Check for RELION 5 subtomogram coordinate columns
    subtomogram_coord_cols = ['rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst']
    legacy_coord_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    
    # Check for RELION 5 subtomogram angle columns
    subtomogram_angle_cols = ['rlnTomoSubtomogramTilt', 'rlnTomoSubtomogramPsi', 'rlnTomoSubtomogramRot']
    legacy_angle_cols = ['rlnAngleTilt', 'rlnAnglePsi', 'rlnAngleRot']

    # Extract coordinates - prefer subtomogram format
    if all(col in df.columns for col in subtomogram_coord_cols):
        centers = np.stack([
            df['rlnCenteredCoordinateXAngst'].astype(float).to_numpy(),
            df['rlnCenteredCoordinateYAngst'].astype(float).to_numpy(),
            df['rlnCenteredCoordinateZAngst'].astype(float).to_numpy(),
        ], axis=1)
    elif all(col in df.columns for col in legacy_coord_cols):
        centers = np.stack([
            df['rlnCoordinateX'].astype(float).to_numpy(),
            df['rlnCoordinateY'].astype(float).to_numpy(),
            df['rlnCoordinateZ'].astype(float).to_numpy(),
        ], axis=1)
    else:
        raise KeyError(f"Missing coordinate columns in STAR file. Available columns: {list(df.columns)}")

    # Extract angles - prefer subtomogram format
    if all(col in df.columns for col in subtomogram_angle_cols):
        eulers = np.stack([
            df['rlnTomoSubtomogramTilt'].astype(float).to_numpy(),
            df['rlnTomoSubtomogramPsi'].astype(float).to_numpy(),
            df['rlnTomoSubtomogramRot'].astype(float).to_numpy(),
        ], axis=1)
    elif all(col in df.columns for col in legacy_angle_cols):
        # Fall back to legacy angle tags
        eulers = np.stack([
            df['rlnAngleTilt'].astype(float).to_numpy(),
            df['rlnAnglePsi'].astype(float).to_numpy(),
            df['rlnAngleRot'].astype(float).to_numpy(),
        ], axis=1)
    else:
        # If angles missing, default to zeros
        eulers = np.zeros((len(df), 3), dtype=float)

    return {'centers': centers, 'eulers': eulers}

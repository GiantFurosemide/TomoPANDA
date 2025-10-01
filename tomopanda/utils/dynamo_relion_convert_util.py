"""
Dynamo-RELION Conversion Utilities for TomoPANDA

This module provides utilities for converting RELION STAR files to Dynamo format,
including .tbl (table) and .vll (volume list) files.

Author: TomoPANDA Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import math

from .relion_utils import read_star, parse_particles_from_star


class DynamoConverter:
    """
    Utility class for converting RELION STAR files to Dynamo format.
    """
    
    @staticmethod
    def relion_to_dynamo_euler(tilt: float, psi: float, rot: float) -> Tuple[float, float, float]:
        """
        Convert RELION Euler angles to Dynamo Euler angles.
        
        RELION uses ZYZ convention: R = Rz(rot) * Ry(tilt) * Rz(psi)
        Dynamo uses ZYZ convention but with different angle definitions.
        
        Args:
            tilt: RELION tilt angle in degrees
            psi: RELION psi angle in degrees  
            rot: RELION rot angle in degrees
            
        Returns:
            Tuple of (phi, theta, psi) in degrees for Dynamo
        """
        # Convert to radians
        tilt_rad = math.radians(tilt)
        psi_rad = math.radians(psi)
        rot_rad = math.radians(rot)
        
        # RELION ZYZ rotation matrix
        Rz_psi = np.array([
            [math.cos(psi_rad), -math.sin(psi_rad), 0],
            [math.sin(psi_rad), math.cos(psi_rad), 0],
            [0, 0, 1]
        ])
        
        Ry_tilt = np.array([
            [math.cos(tilt_rad), 0, math.sin(tilt_rad)],
            [0, 1, 0],
            [-math.sin(tilt_rad), 0, math.cos(tilt_rad)]
        ])
        
        Rz_rot = np.array([
            [math.cos(rot_rad), -math.sin(rot_rad), 0],
            [math.sin(rot_rad), math.cos(rot_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz(rot) * Ry(tilt) * Rz(psi)
        R = Rz_rot @ Ry_tilt @ Rz_psi
        
        # Convert rotation matrix back to Dynamo ZYZ angles
        # For Dynamo, we need to extract phi, theta, psi from the rotation matrix
        # Using the inverse of ZYZ convention
        if abs(R[2, 2]) < 1e-6:
            # Special case when cos(theta) = 0
            phi = math.atan2(R[1, 0], R[0, 0])
            theta = math.pi / 2
            psi = 0.0
        else:
            theta = math.acos(np.clip(R[2, 2], -1.0, 1.0))
            phi = math.atan2(R[2, 1], R[2, 0])
            psi = math.atan2(R[1, 2], -R[0, 2])
        
        # Convert to degrees and ensure proper ranges
        phi_deg = math.degrees(phi)
        theta_deg = math.degrees(theta)
        psi_deg = math.degrees(psi)
        
        # Ensure angles are in Dynamo ranges
        # phi: -180 to +180
        # theta: 0 to 180  
        # psi: -180 to +180
        phi_deg = ((phi_deg + 180) % 360) - 180
        theta_deg = theta_deg % 180
        psi_deg = ((psi_deg + 180) % 360) - 180
        
        return phi_deg, theta_deg, psi_deg
    
    @staticmethod
    def create_tbl_row(tag: int, 
                      coordinates: np.ndarray, 
                      euler_angles: np.ndarray, 
                      tomogram_index: int,
                      model_index: int = 1) -> List[str]:
        """
        Create a single Dynamo .tbl row.
        
        Args:
            tag: Particle tag (column 1)
            coordinates: (x, y, z) coordinates in voxels (columns 24-26)
            euler_angles: (phi, theta, psi) angles in degrees (columns 7-9)
            tomogram_index: Tomogram index in .vll file (column 20)
            model_index: Model index (column 31)
            
        Returns:
            List of 35 strings representing the table row
        """
        x, y, z = coordinates
        phi, theta, psi = euler_angles
        
        # Create 35-column row with defaults
        row = ['0'] * 35
        
        # Column 1: tag (1-based)
        row[0] = str(tag)
        
        # Column 2: aligned (default = "1")
        row[1] = "1"
        
        # Columns 7-9: Euler angles (phi, theta, psi)
        row[6] = f"{phi:.6f}"  # phi
        row[7] = f"{theta:.6f}"  # theta  
        row[8] = f"{psi:.6f}"   # psi
        
        # Column 20: tomogram index (1-based)
        row[19] = str(tomogram_index)
        
        # Columns 24-26: coordinates (x, y, z)
        row[23] = f"{x:.6f}"  # x
        row[24] = f"{y:.6f}"  # y
        row[25] = f"{z:.6f}"  # z
        
        # Column 31: model index (default = "1")
        row[30] = str(model_index)
        
        return row
    
    @staticmethod
    def create_tbl_file(particles_data: Dict[str, np.ndarray],
                       tomogram_names: List[str],
                       output_path: Union[str, Path],
                       start_tag: int = 1,
                       model_index: int = 1) -> None:
        """
        Create Dynamo .tbl file from particle data.
        
        Args:
            particles_data: Dictionary with 'centers' and 'eulers' arrays
            tomogram_names: List of tomogram names (from _rlnTomoName)
            output_path: Output .tbl file path
            start_tag: Starting tag number for particles
            model_index: Model index for all particles
        """
        centers = particles_data['centers']
        eulers = particles_data['eulers']
        
        if len(centers) == 0:
            raise ValueError("No particles to convert")
        
        # Create unique tomogram name to index mapping (preserve order)
        unique_tomograms = list(dict.fromkeys(tomogram_names))  # Preserve order, remove duplicates
        tomogram_to_index = {name: i + 1 for i, name in enumerate(unique_tomograms)}
        
        # Convert RELION angles to Dynamo angles
        dynamo_eulers = []
        for tilt, psi, rot in eulers:
            phi, theta, psi_dynamo = DynamoConverter.relion_to_dynamo_euler(tilt, psi, rot)
            dynamo_eulers.append([phi, theta, psi_dynamo])
        dynamo_eulers = np.array(dynamo_eulers)
        
        # Create table rows
        table_rows = []
        for i, (coords, angles, tomogram_name) in enumerate(zip(centers, dynamo_eulers, tomogram_names)):
            tag = start_tag + i
            tomogram_idx = tomogram_to_index[tomogram_name]
            row = DynamoConverter.create_tbl_row(tag, coords, angles, tomogram_idx, model_index)
            table_rows.append(row)
        
        # Write .tbl file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for row in table_rows:
                f.write(' '.join(row) + '\n')
        
        print(f"Saved {len(table_rows)} particles to Dynamo .tbl file: {output_path}")
    
    @staticmethod
    def create_vll_file(tomogram_names: List[str], 
                       output_path: Union[str, Path]) -> None:
        """
        Create Dynamo .vll file from tomogram names.
        
        Args:
            tomogram_names: List of unique tomogram names
            output_path: Output .vll file path
        """
        # Get unique tomogram names in order
        unique_tomograms = list(dict.fromkeys(tomogram_names))  # Preserve order, remove duplicates
        
        # Write .vll file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for tomogram_name in unique_tomograms:
                f.write(f"{tomogram_name}\n")
        
        print(f"Saved {len(unique_tomograms)} tomograms to Dynamo .vll file: {output_path}")


def relion_star_to_dynamo_tbl_vll(star_file: Union[str, Path],
                                 tbl_output: Union[str, Path],
                                 vll_output: Union[str, Path],
                                 start_tag: int = 1,
                                 model_index: int = 1) -> None:
    """
    Convert RELION STAR file to Dynamo .tbl and .vll files.
    
    Args:
        star_file: Input RELION STAR file path
        tbl_output: Output Dynamo .tbl file path
        vll_output: Output Dynamo .vll file path
        start_tag: Starting tag number for particles
        model_index: Model index for all particles
    """
    # Read RELION STAR file
    df = read_star(star_file)
    particles_data = parse_particles_from_star(df)
    
    # Extract tomogram names
    if 'rlnTomoName' in df.columns:
        tomogram_names = df['rlnTomoName'].tolist()
    else:
        # Fallback: use single tomogram name
        tomogram_names = ['tomogram'] * len(particles_data['centers'])
    
    # Create .tbl file
    DynamoConverter.create_tbl_file(
        particles_data, 
        tomogram_names, 
        tbl_output, 
        start_tag, 
        model_index
    )
    
    # Create .vll file
    DynamoConverter.create_vll_file(tomogram_names, vll_output)


def convert_matrix_to_tbl(matrix: np.ndarray,
                         target_columns: List[int],
                         output_path: Union[str, Path],
                         start_tag: int = 1) -> None:
    """
    Convert an N×M matrix to Dynamo .tbl format by mapping to specified columns.
    
    Args:
        matrix: Input matrix (N, M) where N is number of particles
        target_columns: List of 1-based column indices to map matrix columns to
        output_path: Output .tbl file path
        start_tag: Starting tag number for particles
    """
    if matrix.shape[1] != len(target_columns):
        raise ValueError(f"Matrix has {matrix.shape[1]} columns but {len(target_columns)} target columns specified")
    
    # Create table rows
    table_rows = []
    for i, row_data in enumerate(matrix):
        # Create 35-column row with defaults
        row = ['0'] * 35
        
        # Column 1: tag
        row[0] = str(start_tag + i)
        
        # Column 2: aligned (default = "1")
        row[1] = "1"
        
        # Map matrix columns to target columns
        for j, col_idx in enumerate(target_columns):
            if 1 <= col_idx <= 35:
                row[col_idx - 1] = f"{row_data[j]:.6f}"
        
        table_rows.append(row)
    
    # Write .tbl file
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        for row in table_rows:
            f.write(' '.join(row) + '\n')
    
    print(f"Saved {len(table_rows)} particles to Dynamo .tbl file: {output_path}")


# Convenience functions for common conversions
def convert_relion_to_dynamo(star_file: Union[str, Path],
                           output_dir: Union[str, Path],
                           base_name: str = "particles",
                           start_tag: int = 1,
                           model_index: int = 1) -> Tuple[Path, Path]:
    """
    Convert RELION STAR file to Dynamo format with automatic file naming.
    
    Args:
        star_file: Input RELION STAR file path
        output_dir: Output directory
        base_name: Base name for output files
        start_tag: Starting tag number for particles
        model_index: Model index for all particles
        
    Returns:
        Tuple of (tbl_file_path, vll_file_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tbl_path = output_dir / f"{base_name}.tbl"
    vll_path = output_dir / f"{base_name}.vll"
    
    relion_star_to_dynamo_tbl_vll(star_file, tbl_path, vll_path, start_tag, model_index)
    
    return tbl_path, vll_path

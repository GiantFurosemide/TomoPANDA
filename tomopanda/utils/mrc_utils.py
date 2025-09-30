"""
MRC File Utilities for TomoPANDA

This module provides utilities for reading and writing MRC files,
specifically for membrane masks and tomogram data.

Author: TomoPANDA Team
"""

import numpy as np
from typing import Tuple, Optional, Union
import mrcfile
from pathlib import Path


class MRCReader:
    """
    Utility class for reading MRC files with proper error handling.
    """
    
    @staticmethod
    def read_mrc(filepath: Union[str, Path], 
                 permissive: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Read MRC file and return data with metadata.
        
        Args:
            filepath: Path to MRC file
            permissive: Whether to use permissive mode for reading
            
        Returns:
            Tuple of (data, metadata) where:
            - data: 3D numpy array
            - metadata: Dictionary containing header information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"MRC file not found: {filepath}")
        
        try:
            with mrcfile.open(str(filepath), permissive=permissive) as mrc:
                data = mrc.data.copy()
                
                # Extract relevant metadata
                metadata = {
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'voxel_size': mrc.voxel_size,
                    'origin': mrc.header.origin,
                    'cell_angles': mrc.header.cella,
                    'cell_dimensions': mrc.header.cellb,
                    'spacegroup': mrc.header.ispg,
                    'mode': mrc.header.mode,
                    'nx': mrc.header.nx,
                    'ny': mrc.header.ny,
                    'nz': mrc.header.nz,
                    'mx': mrc.header.mx,
                    'my': mrc.header.my,
                    'mz': mrc.header.mz,
                    'xlen': mrc.header.cella.x,
                    'ylen': mrc.header.cella.y,
                    'zlen': mrc.header.cella.z,
                    'alpha': mrc.header.cellb.alpha,
                    'beta': mrc.header.cellb.beta,
                    'gamma': mrc.header.cellb.gamma
                }
                
                return data, metadata
                
        except Exception as e:
            raise RuntimeError(f"Failed to read MRC file {filepath}: {str(e)}")
    
    @staticmethod
    def read_membrane_mask(filepath: Union[str, Path]) -> np.ndarray:
        """
        Read membrane mask from MRC file.
        
        Args:
            filepath: Path to membrane mask MRC file
            
        Returns:
            Binary mask array (0/1) indicating membrane regions
        """
        data, _ = MRCReader.read_mrc(filepath)
        
        # Convert to binary mask (0/1)
        if data.dtype != np.uint8:
            # Normalize to 0-1 range and convert to binary
            data = data.astype(float)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            data = (data > 0.5).astype(np.uint8)
        else:
            # Assume already binary, ensure 0/1 values
            data = (data > 0).astype(np.uint8)
        
        return data
    
    @staticmethod
    def read_tomogram(filepath: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Read tomogram from MRC file.
        
        Args:
            filepath: Path to tomogram MRC file
            
        Returns:
            Tuple of (tomogram_data, metadata)
        """
        return MRCReader.read_mrc(filepath)


class MRCWriter:
    """
    Utility class for writing MRC files.
    """
    
    @staticmethod
    def write_mrc(data: np.ndarray, 
                  filepath: Union[str, Path],
                  voxel_size: Optional[Tuple[float, float, float]] = None,
                  origin: Optional[Tuple[float, float, float]] = None,
                  overwrite: bool = True) -> None:
        """
        Write numpy array to MRC file.
        
        Args:
            data: 3D numpy array to write
            filepath: Output file path
            voxel_size: Voxel size (x, y, z) in Angstroms
            origin: Origin coordinates (x, y, z)
            overwrite: Whether to overwrite existing file
        """
        filepath = Path(filepath)
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")
        
        # Ensure data is 3D
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D, got {data.ndim}D")
        
        try:
            with mrcfile.new(str(filepath), overwrite=overwrite) as mrc:
                mrc.set_data(data.astype(np.float32))
                
                # Set voxel size if provided
                if voxel_size is not None:
                    mrc.voxel_size = voxel_size
                
                # Set origin if provided
                if origin is not None:
                    mrc.header.origin = origin
                    
        except Exception as e:
            raise RuntimeError(f"Failed to write MRC file {filepath}: {str(e)}")
    
    @staticmethod
    def write_membrane_mask(mask: np.ndarray, 
                           filepath: Union[str, Path],
                           voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Write membrane mask to MRC file.
        
        Args:
            mask: Binary mask array (0/1)
            filepath: Output file path
            voxel_size: Voxel size (x, y, z) in Angstroms
        """
        # Ensure mask is binary
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("Mask must contain only 0 and 1 values")
        
        MRCWriter.write_mrc(mask.astype(np.uint8), filepath, voxel_size)
    
    @staticmethod
    def write_tomogram(tomogram: np.ndarray, 
                      filepath: Union[str, Path],
                      voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Write tomogram to MRC file.
        
        Args:
            tomogram: Tomogram data array
            filepath: Output file path
            voxel_size: Voxel size (x, y, z) in Angstroms
        """
        MRCWriter.write_mrc(tomogram, filepath, voxel_size)


def load_membrane_mask(filepath: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to load membrane mask.
    
    Args:
        filepath: Path to membrane mask MRC file
        
    Returns:
        Binary mask array (0/1)
    """
    return MRCReader.read_membrane_mask(filepath)


def load_tomogram(filepath: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to load tomogram.
    
    Args:
        filepath: Path to tomogram MRC file
        
    Returns:
        Tuple of (tomogram_data, metadata)
    """
    return MRCReader.read_tomogram(filepath)


def save_coordinates(centers: np.ndarray, 
                    normals: np.ndarray, 
                    filepath: Union[str, Path],
                    voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
    """
    Save sampling coordinates to CSV file for subtomogram processing.
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3) - membrane surface normals
        filepath: Output file path
        voxel_size: Voxel size for coordinate scaling
    """
    if len(centers) == 0:
        raise ValueError("No coordinates to save")
    
    # Scale coordinates if voxel size is provided
    if voxel_size is not None:
        centers_scaled = centers * np.array(voxel_size)
    else:
        centers_scaled = centers
    
    # Save as CSV with coordinates and normals for subtomogram processing
    import pandas as pd
    
    data = np.column_stack([centers_scaled, normals])
    columns = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)
    
    print(f"Saved {len(centers)} sampling points to {filepath}")


def save_subtomogram_coordinates(centers: np.ndarray, 
                                normals: np.ndarray, 
                                filepath: Union[str, Path],
                                tomogram_name: str = "tomogram",
                                particle_diameter: float = 200.0,
                                voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
    """
    Save sampling coordinates in RELION 5 subtomogram format.
    
    This function creates a STAR file compatible with RELION 5 subtomogram averaging,
    using the rlnTomoSubtomogramRot/Tilt/Psi tags for membrane orientation.
    
    Args:
        centers: Sample centers (K, 3)
        normals: Sample normals (K, 3) - membrane surface normals
        filepath: Output STAR file path
        tomogram_name: Name of the tomogram
        particle_diameter: Particle diameter in Angstroms
        voxel_size: Voxel size for coordinate scaling
    """
    if len(centers) == 0:
        raise ValueError("No coordinates to save")
    
    # Import here to avoid circular imports
    from .relion_utils import RELIONConverter
    
    # Scale coordinates if voxel size is provided
    if voxel_size is not None:
        centers_scaled = centers * np.array(voxel_size)
    else:
        centers_scaled = centers
    
    # Convert membrane normals to Euler angles for subtomogram orientation
    euler_angles = []
    for normal in normals:
        tilt, psi, rot = RELIONConverter.normal_to_euler(normal)
        euler_angles.append([tilt, psi, rot])
    
    euler_angles = np.array(euler_angles)
    
    # Create STAR file data with only requested columns
    import pandas as pd

    df = pd.DataFrame({
        'rlnCoordinateX': centers_scaled[:, 0],
        'rlnCoordinateY': centers_scaled[:, 1],
        'rlnCoordinateZ': centers_scaled[:, 2],
        'rlnTomoSubtomogramRot': euler_angles[:, 2],
        'rlnTomoSubtomogramTilt': euler_angles[:, 0],
        'rlnTomoSubtomogramPsi': euler_angles[:, 1],
        'rlnTomoName': [tomogram_name] * len(centers),
        'rlnTomoParticleId': list(range(len(centers))),
    })

    # Write simple RELION loop with only these columns
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        f.write("data_\n\n")
        f.write("loop_\n")
        for col in df.columns:
            f.write(f"_{col}\n")
        for _, row in df.iterrows():
            vals = []
            for val in row.values:
                if isinstance(val, float):
                    vals.append(f"{val:.6f}")
                else:
                    vals.append(str(val))
            f.write(" ".join(vals) + "\n")

    print(f"Saved {len(centers)} subtomogram particles to RELION STAR file: {filepath}")

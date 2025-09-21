"""
Sample command for TomoPANDA CLI

This module provides sampling commands for particle picking,
including mesh geodesic sampling for membrane-based particle detection.

Author: TomoPANDA Team
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from .base import BaseCommand


class SampleCommand(BaseCommand):
    """
    Sample command for particle picking operations.
    """
    
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(
            description="Sample particles from tomograms",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Mesh geodesic sampling with synthetic data
  tomopanda sample mesh-geodesic --create-synthetic --output results/
  
  # Mesh geodesic sampling with real membrane mask
  tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/
  
  # Custom parameters
  tomopanda sample mesh-geodesic --mask mask.mrc --min-distance 25.0 --particle-radius 12.0
            """
        )
        
        # Add subcommands
        subparsers = self.parser.add_subparsers(
            dest='method',
            help='Sampling method to use',
            required=True
        )
        
        # Mesh geodesic subcommand
        mesh_parser = subparsers.add_parser(
            'mesh-geodesic',
            help='Mesh geodesic sampling for membrane-based particle picking',
            description='Perform mesh geodesic sampling on membrane masks to generate particle picking candidates'
        )
        
        self._setup_mesh_geodesic_parser(mesh_parser)
    
    def _setup_mesh_geodesic_parser(self, parser: argparse.ArgumentParser):
        """Setup arguments for mesh geodesic sampling."""
        
        # Input options
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            '--mask',
            type=str,
            help='Path to membrane mask MRC file'
        )
        input_group.add_argument(
            '--create-synthetic',
            action='store_true',
            help='Create synthetic membrane mask for testing'
        )
        
        # Output options
        parser.add_argument(
            '--output',
            type=str,
            default='sampling_results',
            help='Output directory for results (default: sampling_results)'
        )
        
        # Sampling parameters
        parser.add_argument(
            '--min-distance',
            type=float,
            default=20.0,
            help='Minimum distance between sampling points in pixels (default: 20.0)'
        )
        parser.add_argument(
            '--particle-radius',
            type=float,
            default=10.0,
            help='Particle radius for boundary checking in pixels (default: 10.0)'
        )
        parser.add_argument(
            '--smoothing-sigma',
            type=float,
            default=1.5,
            help='Gaussian smoothing sigma (default: 1.5)'
        )
        parser.add_argument(
            '--taubin-iterations',
            type=int,
            default=10,
            help='Number of Taubin smoothing iterations (default: 10)'
        )
        
        # Synthetic data parameters
        parser.add_argument(
            '--synthetic-shape',
            type=int,
            nargs=3,
            default=[100, 100, 100],
            metavar=('Z', 'Y', 'X'),
            help='Shape for synthetic membrane mask (default: 100 100 100)'
        )
        parser.add_argument(
            '--synthetic-center',
            type=int,
            nargs=3,
            default=[50, 50, 50],
            metavar=('Z', 'Y', 'X'),
            help='Center for synthetic membrane mask (default: 50 50 50)'
        )
        parser.add_argument(
            '--synthetic-radius',
            type=int,
            default=30,
            help='Radius for synthetic membrane mask (default: 30)'
        )
        
        # Output format options
        parser.add_argument(
            '--tomogram-name',
            type=str,
            default='tomogram',
            help='Name of the tomogram for RELION output (default: tomogram)'
        )
        parser.add_argument(
            '--particle-diameter',
            type=float,
            default=200.0,
            help='Particle diameter in Angstroms (default: 200.0)'
        )
        parser.add_argument(
            '--confidence',
            type=float,
            default=0.5,
            help='Confidence score for particles (default: 0.5)'
        )
        
        # Advanced options
        parser.add_argument(
            '--voxel-size',
            type=float,
            nargs=3,
            metavar=('X', 'Y', 'Z'),
            help='Voxel size in Angstroms (X, Y, Z)'
        )
        parser.add_argument(
            '--sigma-tilt',
            type=float,
            default=30.0,
            help='Standard deviation for tilt angle prior (default: 30.0)'
        )
        parser.add_argument(
            '--sigma-psi',
            type=float,
            default=30.0,
            help='Standard deviation for psi angle prior (default: 30.0)'
        )
        parser.add_argument(
            '--sigma-rot',
            type=float,
            default=30.0,
            help='Standard deviation for rot angle prior (default: 30.0)'
        )
        
        # Visualization options
        parser.add_argument(
            '--create-visualization',
            action='store_true',
            help='Create visualization script for results'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the sample command."""
        
        if args.method == 'mesh-geodesic':
            return self._run_mesh_geodesic(args)
        else:
            self.parser.error(f"Unknown sampling method: {args.method}")
            return 1
    
    def _run_mesh_geodesic(self, args: argparse.Namespace) -> int:
        """Run mesh geodesic sampling."""
        
        try:
            # Import required modules
            from tomopanda.core.mesh_geodesic import create_mesh_geodesic_sampler
            from tomopanda.utils.mrc_utils import MRCReader, MRCWriter, load_membrane_mask
            from tomopanda.utils.relion_utils import (
                convert_to_relion_star, 
                convert_to_coordinate_file, 
                convert_to_prior_angles
            )
            
        except ImportError as e:
            print(f"Error: Missing required dependencies: {e}")
            print("Please install mesh geodesic dependencies:")
            print("  python install_mesh_geodesic_deps.py")
            return 1
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=== Mesh Geodesic Sampling for CryoET Particle Picking ===")
        print(f"Output directory: {output_dir}")
        
        # Load or create membrane mask
        if args.create_synthetic:
            print("Creating synthetic membrane mask...")
            mask = self._create_synthetic_mask(args)
            mask_path = output_dir / "synthetic_mask.mrc"
            
            # Save synthetic mask
            MRCWriter.write_membrane_mask(mask, mask_path)
            print(f"Saved synthetic mask to: {mask_path}")
            
        else:
            print(f"Loading membrane mask from: {args.mask}")
            mask = load_membrane_mask(args.mask)
        
        print(f"Membrane mask shape: {mask.shape}")
        print(f"Membrane volume fraction: {mask.sum() / mask.size:.3f}")
        
        # Create mesh geodesic sampler
        print("\n=== Creating Mesh Geodesic Sampler ===")
        sampler = create_mesh_geodesic_sampler(
            min_distance=args.min_distance,
            smoothing_sigma=args.smoothing_sigma,
            taubin_iterations=args.taubin_iterations
        )
        
        if args.verbose:
            print(f"Sampling parameters:")
            print(f"  - Minimum distance: {args.min_distance}")
            print(f"  - Smoothing sigma: {args.smoothing_sigma}")
            print(f"  - Taubin iterations: {args.taubin_iterations}")
            print(f"  - Particle radius: {args.particle_radius}")
        
        # Perform mesh geodesic sampling
        print("\n=== Performing Mesh Geodesic Sampling ===")
        try:
            centers, normals = sampler.sample_membrane_points(
                mask, 
                particle_radius=args.particle_radius
            )
            
            print(f"Sampling completed successfully!")
            print(f"Number of sampling points: {len(centers)}")
            
            if len(centers) > 0:
                if args.verbose:
                    print(f"Coordinate ranges:")
                    print(f"  X: {centers[:, 0].min():.1f} to {centers[:, 0].max():.1f}")
                    print(f"  Y: {centers[:, 1].min():.1f} to {centers[:, 1].max():.1f}")
                    print(f"  Z: {centers[:, 2].min():.1f} to {centers[:, 2].max():.1f}")
                    
                    print(f"Normal vector statistics:")
                    print(f"  Mean length: {np.linalg.norm(normals, axis=1).mean():.3f}")
                    print(f"  Std length: {np.linalg.norm(normals, axis=1).std():.3f}")
            
        except Exception as e:
            print(f"Error during sampling: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        
        # Save results in multiple formats
        print("\n=== Saving Results ===")
        
        if len(centers) > 0:
            # Save coordinates as CSV
            coord_file = output_dir / "sampling_coordinates.csv"
            self._save_coordinates_csv(centers, normals, coord_file)
            
            # Save RELION STAR file
            star_file = output_dir / "particles.star"
            convert_to_relion_star(
                centers, normals, star_file, 
                tomogram_name=args.tomogram_name,
                particle_diameter=args.particle_diameter
            )
            
            # Save coordinate file
            coord_file_relion = output_dir / "coordinates.csv"
            convert_to_coordinate_file(centers, normals, coord_file_relion, args.voxel_size)
            
            # Save prior angles file
            prior_file = output_dir / "prior_angles.csv"
            convert_to_prior_angles(
                centers, normals, prior_file,
                sigma_tilt=args.sigma_tilt,
                sigma_psi=args.sigma_psi,
                sigma_rot=args.sigma_rot
            )
            
            print(f"Results saved to:")
            print(f"  - Coordinates: {coord_file}")
            print(f"  - RELION STAR: {star_file}")
            print(f"  - RELION coordinates: {coord_file_relion}")
            print(f"  - Prior angles: {prior_file}")
            
            # Create visualization script if requested
            if args.create_visualization:
                vis_script = output_dir / "visualize_results.py"
                self._create_visualization_script(vis_script, centers, normals)
                print(f"  - Visualization script: {vis_script}")
            
        else:
            print("No sampling points found. Check your membrane mask.")
            return 1
        
        print("\n=== Sampling Complete ===")
        print(f"Total sampling points: {len(centers)}")
        print(f"Results saved to: {output_dir}")
        
        return 0
    
    def _create_synthetic_mask(self, args: argparse.Namespace) -> np.ndarray:
        """Create synthetic membrane mask for testing."""
        shape = tuple(args.synthetic_shape)
        center = tuple(args.synthetic_center)
        radius = args.synthetic_radius
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Create sphere
        distance = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
        mask = (distance <= radius).astype(np.uint8)
        
        return mask
    
    def _save_coordinates_csv(self, centers: np.ndarray, normals: np.ndarray, filepath: Path):
        """Save coordinates as CSV file."""
        import pandas as pd
        
        data = np.column_stack([centers, normals])
        columns = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        
        print(f"Saved {len(centers)} sampling points to {filepath}")
    
    def _create_visualization_script(self, script_path: Path, centers: np.ndarray, normals: np.ndarray):
        """Create visualization script for results."""
        script_content = f'''#!/usr/bin/env python3
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
ax1.set_title(f'{{len(centers)}} Sampling Points')

# Plot 2: Normal vectors
ax2 = fig.add_subplot(122, projection='3d')
# Show subset of normals for clarity
step = max(1, len(centers) // 50)
for i in range(0, len(centers), step):
    center = centers[i]
    normal = normals[i] * 5  # Scale for visibility
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

print(f"Visualization saved as 'sampling_visualization.png'")
print(f"Total sampling points: {{len(centers)}}")
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)

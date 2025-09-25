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

    def get_name(self) -> str:
        return "sample"

    def get_description(self) -> str:
        return "Sampling utilities (e.g., mesh-geodesic) for particle picking"

    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add the 'sample' command and its subcommands to the CLI."""
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description="Sampling utilities for particle picking",
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

        # Add subcommands for sampling methods
        method_subparsers = parser.add_subparsers(
            dest='method',
            help='Sampling method to use',
            required=True
        )

        # Mesh geodesic subcommand
        mesh_parser = method_subparsers.add_parser(
            'mesh-geodesic',
            help='Mesh geodesic sampling for membrane-based particle picking',
            description='Perform mesh geodesic sampling on membrane masks to generate particle picking candidates'
        )

        self._setup_mesh_geodesic_parser(mesh_parser)

        # Do not add common args here to avoid conflicts with existing --output/--verbose
        return parser

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the selected sampling method."""
        if getattr(args, 'method', None) == 'mesh-geodesic':
            return self._run_mesh_geodesic(args)
        print(f"Unknown sampling method: {getattr(args, 'method', None)}")
        return 1

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
            '--save-intermediates',
            action='store_true',
            help='Save intermediate MRC volumes: surface mask and rasterized mesh surface'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
    
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
            if args.verbose:
                print("Euler angles are derived from normals (tilt, psi, rot=0) per RELION convention.")

            # Save intermediates if requested
            if args.save_intermediates:
                from tomopanda.core.mesh_geodesic import MeshGeodesicSampler
                from tomopanda.utils.mrc_utils import MRCWriter
                print("\n=== Saving Intermediate Volumes (surface & mesh) ===")
                surface = MeshGeodesicSampler.extract_surface_voxels(mask)
                surface_path = output_dir / "surface_mask.mrc"
                MRCWriter.write_membrane_mask(surface, surface_path)
                # Recompute SDF and mesh to rasterize surface mesh
                phi = sampler.create_signed_distance_field(mask)
                mesh = sampler.extract_mesh_from_sdf(phi)
                mesh_vol = MeshGeodesicSampler.rasterize_mesh_to_volume(mesh, mask.shape)
                mesh_path = output_dir / "mesh_surface.mrc"
                MRCWriter.write_membrane_mask(mesh_vol, mesh_path)
                print(f"  - Surface mask: {surface_path}")
                print(f"  - Mesh surface: {mesh_path}")
            
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
        """Create synthetic membrane mask for testing (delegates to core utility)."""
        from tomopanda.core.mesh_geodesic import generate_synthetic_mask
        return generate_synthetic_mask(
            shape=tuple(args.synthetic_shape),
            center=tuple(args.synthetic_center),
            radius=args.synthetic_radius,
        )
    
    def _save_coordinates_csv(self, centers: np.ndarray, normals: np.ndarray, filepath: Path):
        """Deprecated local writer; use core.save_sampling_outputs instead."""
        from tomopanda.core.mesh_geodesic import save_sampling_outputs
        out_dir = filepath.parent
        save_sampling_outputs(
            out_dir,
            centers,
            normals,
            tomogram_name='tomogram',
            particle_diameter=200.0,
            create_vis_script=False,
        )
    
    def _create_visualization_script(self, script_path: Path, centers: np.ndarray, normals: np.ndarray):
        """Delegate viz script creation to core helper."""
        from tomopanda.core.mesh_geodesic import create_visualization_script
        create_visualization_script(script_path, centers, normals)

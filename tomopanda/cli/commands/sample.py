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
  # Extract all triangle centers with synthetic data
  tomopanda sample mesh-geodesic --create-synthetic --output results/

  # Extract triangle centers with real membrane mask
  tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

  # Custom parameters with mesh density control
  tomopanda sample mesh-geodesic --mask mask.mrc --expected-particle-size 20.0 --random-seed 42

  # Generate different mesh variants
  tomopanda sample mesh-geodesic --create-synthetic --random-seed 42 --output variant_1
  tomopanda sample mesh-geodesic --create-synthetic --random-seed 123 --output variant_2
            """
        )

        # Add subcommands for sampling methods
        method_subparsers = parser.add_subparsers(
            dest='method',
            help='Sampling method to use',
            required=True
        )

        # Mesh geodesic subcommand (triangle extraction only)
        mesh_parser = method_subparsers.add_parser(
            'mesh-geodesic',
            help='Extract all triangle centers and normals from membrane mesh',
            description='Extract all triangle centers and normals from membrane masks without distance-based sampling.'
        )

        self._setup_mesh_geodesic_parser(mesh_parser)

        # Voxel-based sampler subcommand
        voxel_parser = method_subparsers.add_parser(
            'voxel-sample',
            help='Voxel-based surface sampling producing RELION 5 particle STAR',
            description='Extract membrane surface voxels, sample with min-distance, and write RELION 5 STAR'
        )
        self._setup_voxel_sample_parser(voxel_parser)

        # Do not add common args here to avoid conflicts with existing --output/--verbose
        return parser

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the selected sampling method."""
        if getattr(args, 'method', None) == 'mesh-geodesic':
            return self._run_mesh_geodesic(args)
        if getattr(args, 'method', None) == 'voxel-sample':
            return self._run_voxel_sample(args)
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
        
        # Mode is now fixed to triangles only
        # No mode selection needed
        
        # Mesh parameters (no distance-based sampling)
        # Removed min-distance and particle-radius as they are not needed for triangle extraction
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
            help='Number of Taubin smoothing iterations (default: 10). Mutually exclusive with --expected-particle-size'
        )
        parser.add_argument(
            '--expected-particle-size',
            type=float,
            help='Expected particle size in pixels for mesh density control. Automatically calculates taubin iterations. Mutually exclusive with --taubin-iterations'
        )
        parser.add_argument(
            '--random-seed',
            type=int,
            help='Random seed for mesh generation (None for deterministic)'
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
    
    def _setup_voxel_sample_parser(self, parser: argparse.ArgumentParser):
        """Setup arguments for voxel-based surface sampling."""
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
            default='voxel_sampling_results',
            help='Output directory for results (default: voxel_sampling_results)'
        )

        # Sampling parameters
        parser.add_argument(
            '--min-distance',
            type=float,
            default=20.0,
            help='Minimum distance between sampling points in pixels (default: 20.0)'
        )
        parser.add_argument(
            '--edge-distance',
            type=float,
            default=10.0,
            help='Edge margin in pixels; avoid sampling within this distance from volume boundary'
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
            '--voxel-size',
            type=float,
            nargs=3,
            metavar=('X', 'Y', 'Z'),
            help='Voxel size in Angstroms (X, Y, Z) applied to coordinates'
        )

        # Visualization and intermediates
        parser.add_argument(
            '--save-intermediates',
            action='store_true',
            help='Save intermediate MRC volumes: surface mask'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
    
    def _setup_triangle_centers_parser(self, parser: argparse.ArgumentParser):
        """Setup arguments for triangle centers extraction."""
        
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
            default='triangle_centers_results',
            help='Output directory for results (default: triangle_centers_results)'
        )
        
        # Mesh parameters
        parser.add_argument(
            '--expected-particle-size',
            type=float,
            help='Expected particle size in pixels for mesh density control'
        )
        parser.add_argument(
            '--random-seed',
            type=int,
            help='Random seed for mesh generation (None for deterministic)'
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
        
        # Advanced options
        parser.add_argument(
            '--voxel-size',
            type=float,
            nargs=3,
            metavar=('X', 'Y', 'Z'),
            help='Voxel size in Angstroms (X, Y, Z)'
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

    def _run_voxel_sample(self, args: argparse.Namespace) -> int:
        """Run voxel-based surface sampling and export to RELION 5 STAR."""
        try:
            from tomopanda.core.voxel_sample import (
                info_extract,
                sample as voxel_sample_run,
                save_field_as_relion_star,
                extract_centers_and_normals_from_field,
            )
            from tomopanda.utils.mrc_utils import MRCWriter, load_membrane_mask
            from tomopanda.utils.relion_utils import (
                convert_to_coordinate_file,
                convert_to_prior_angles,
            )
        except ImportError as e:
            print(f"Error: Missing required dependencies: {e}")
            return 1

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=== Voxel-based Surface Sampling ===")
        print(f"Output directory: {output_dir}")

        # Load or create membrane mask
        if args.create_synthetic:
            print("Creating synthetic membrane mask...")
            try:
                from tomopanda.core.mesh_geodesic import generate_synthetic_mask
                mask = generate_synthetic_mask(
                    shape=tuple(args.synthetic_shape),
                    center=tuple(args.synthetic_center),
                    radius=args.synthetic_radius,
                )
            except Exception as e:
                print(f"Failed to create synthetic mask: {e}")
                return 1
            mask_path = output_dir / "synthetic_mask.mrc"
            MRCWriter.write_membrane_mask(mask, mask_path)
            print(f"Saved synthetic mask to: {mask_path}")
        else:
            print(f"Loading membrane mask from: {args.mask}")
            mask = load_membrane_mask(args.mask)

        print(f"Membrane mask shape: {mask.shape}")
        print(f"Membrane volume fraction: {mask.sum() / mask.size:.3f}")

        # Extract surface and orientations
        print("\n=== Extracting Surface and Orientations ===")
        surface_mask, orientations = info_extract(mask)
        if args.save_intermediates:
            surface_path = output_dir / "surface_mask.mrc"
            MRCWriter.write_membrane_mask(surface_mask, surface_path)
            print(f"  - Surface mask: {surface_path}")

        # Run voxel sampling
        print("\n=== Sampling Surface Voxels ===")
        try:
            field = voxel_sample_run(
                min_distance=args.min_distance,
                edge_distance=args.edge_distance,
                surface_mask=surface_mask,
                orientations=orientations,
            )
        except Exception as e:
            print(f"Error during voxel sampling: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        centers, normals = extract_centers_and_normals_from_field(field)
        print(f"Sampling completed successfully with {len(centers)} points.")

        # Save outputs
        print("\n=== Saving Results ===")
        if len(centers) == 0:
            print("No sampling points found. Check your membrane mask or parameters.")
            return 1

        # Determine tomogram name: prefer explicit --tomogram-name, else use full --mask path
        tomogram_name_value = (
            args.tomogram_name if getattr(args, 'tomogram_name', None) and args.tomogram_name != 'tomogram'
            else (args.mask if getattr(args, 'mask', None) else 'tomogram')
        )

        # STAR file
        star_file = output_dir / "particles.star"
        save_field_as_relion_star(
            field,
            star_file,
            tomogram_name=tomogram_name_value,
            particle_diameter=args.particle_diameter,
            voxel_size=tuple(args.voxel_size) if args.voxel_size else None,
        )

        # Coordinate CSV
        coord_file = output_dir / "coordinates.csv"
        convert_to_coordinate_file(
            centers,
            normals,
            coord_file,
            tuple(args.voxel_size) if args.voxel_size else None,
        )

        # Prior angles CSV
        prior_file = output_dir / "prior_angles.csv"
        convert_to_prior_angles(
            centers,
            normals,
            prior_file,
        )

        print(f"  - RELION STAR: {star_file}")
        print(f"  - Coordinates: {coord_file}")
        print(f"  - Prior angles: {prior_file}")

        print("\n=== Voxel Sampling Complete ===")
        print(f"Total sampling points: {len(centers)}")
        print(f"Results saved to: {output_dir}")
        return 0
    
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
        
        # Validate mutually exclusive parameters
        if getattr(args, 'expected_particle_size', None) is not None and args.taubin_iterations != 10:
            print("Warning: --expected-particle-size and --taubin-iterations are mutually exclusive.")
            print("When --expected-particle-size is specified, taubin iterations are automatically calculated.")
            print("Ignoring --taubin-iterations and using auto-calculated value.")
        
        # Create mesh geodesic sampler
        print("\n=== Creating Mesh Geodesic Sampler ===")
        sampler = create_mesh_geodesic_sampler(
            smoothing_sigma=args.smoothing_sigma,
            taubin_iterations=args.taubin_iterations,
            expected_particle_size=getattr(args, 'expected_particle_size', None),
            random_seed=getattr(args, 'random_seed', None)
        )
        
        if args.verbose:
            print(f"Triangle extraction parameters:")
            print(f"  - Smoothing sigma: {args.smoothing_sigma}")
            print(f"  - Taubin iterations: {args.taubin_iterations}")
            if getattr(args, 'expected_particle_size', None) is not None:
                print(f"  - Expected particle size: {args.expected_particle_size}")
            if getattr(args, 'random_seed', None) is not None:
                print(f"  - Random seed: {args.random_seed}")
        
        # Extract all triangle centers and normals
        print("\n=== Extracting All Triangle Centers ===")
        try:
            triangle_data = sampler.get_all_triangle_centers_and_normals(mask)
            centers = triangle_data[:, :3]  # x, y, z
            normals = triangle_data[:, 3:]  # nx, ny, nz
            print(f"Extracted {len(triangle_data)} triangle centers")
        except Exception as e:
            print(f"Error during triangle extraction: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

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
            self._save_coordinates_csv(centers, normals, coord_file, 
                                     tomogram_name=(args.tomogram_name if getattr(args, 'tomogram_name', None) and args.tomogram_name != 'tomogram' else (args.mask if getattr(args, 'mask', None) else 'tomogram')))
            
            # Save RELION STAR file (simplified format)
            star_file = output_dir / "particles.star"
            from tomopanda.core.mesh_geodesic import _save_simplified_relion_star
            _save_simplified_relion_star(
                centers, normals, star_file, 
                tomogram_name=(args.tomogram_name if getattr(args, 'tomogram_name', None) and args.tomogram_name != 'tomogram' else (args.mask if getattr(args, 'mask', None) else 'tomogram')),
                particle_diameter=args.particle_diameter,
                voxel_size=args.voxel_size
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
    
    def _save_coordinates_csv(self, centers: np.ndarray, normals: np.ndarray, filepath: Path, tomogram_name: str = 'tomogram'):
        """Deprecated local writer; use core.save_sampling_outputs instead."""
        from tomopanda.core.mesh_geodesic import save_sampling_outputs
        out_dir = filepath.parent
        save_sampling_outputs(
            out_dir,
            centers,
            normals,
            tomogram_name=tomogram_name,
            particle_diameter=200.0,
            create_vis_script=False,
        )
    
    def _create_visualization_script(self, script_path: Path, centers: np.ndarray, normals: np.ndarray):
        """Delegate viz script creation to core helper."""
        from tomopanda.core.mesh_geodesic import create_visualization_script
        create_visualization_script(script_path, centers, normals)

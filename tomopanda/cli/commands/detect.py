"""
Particle detection command
tomopanda detect <input> [options]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .base import BaseCommand


class DetectCommand(BaseCommand):
    """Particle detection command"""
    
    def get_name(self) -> str:
        return "detect"
    
    def get_description(self) -> str:
        return "Detect membrane protein particles in tomographic data"
    
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description="Detect membrane protein particles in tomographic data using SE(3) equivariant transformer"
        )
        
        # Required parameters
        parser.add_argument(
            'input',
            type=str,
            help='Input tomographic data file path'
        )
        
        # Detection parameters
        parser.add_argument(
            '--model', '-m',
            type=str,
            default='default',
            help='Detection model to use (default: default)'
        )
        parser.add_argument(
            '--threshold', '-t',
            type=float,
            default=0.5,
            help='Detection confidence threshold (default: 0.5)'
        )
        parser.add_argument(
            '--min-size',
            type=int,
            default=10,
            help='Minimum particle size (default: 10)'
        )
        parser.add_argument(
            '--max-size',
            type=int,
            default=100,
            help='Maximum particle size (default: 100)'
        )
        
        # Output parameters
        parser.add_argument(
            '--format', '-f',
            choices=['json', 'csv', 'pdb', 'mrc'],
            default='json',
            help='Output format (default: json)'
        )
        parser.add_argument(
            '--save-visualization',
            action='store_true',
            help='Save detection result visualization images'
        )
        
        # Processing parameters
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help='Batch size (default: 1)'
        )
        parser.add_argument(
            '--device',
            choices=['cpu', 'cuda', 'auto'],
            default='auto',
            help='Computing device (default: auto)'
        )
        
        # Add common arguments
        self.add_common_args(parser)
        
        return parser
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute detection command"""
        try:
            # Validate input file
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file does not exist: {args.input}")
                return 1
            
            # Set output path
            output_path = self._get_output_path(args)
            
            # Execute detection
            print(f"Starting particle detection...")
            print(f"Input file: {args.input}")
            print(f"Output path: {output_path}")
            print(f"Model: {args.model}")
            print(f"Threshold: {args.threshold}")
            
            # TODO: Implement actual detection logic
            # This should call TomoPANDA core detection module
            self._run_detection(args, output_path)
            
            print("Detection completed!")
            return 0
            
        except Exception as e:
            print(f"Error occurred during detection: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _get_output_path(self, args: argparse.Namespace) -> Path:
        """Get output path"""
        if args.output:
            return Path(args.output)
        
        input_path = Path(args.input)
        output_dir = input_path.parent / f"{input_path.stem}_detected"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def _run_detection(self, args: argparse.Namespace, output_path: Path) -> None:
        """Run detection logic"""
        # TODO: Implement actual detection logic
        # This should call TomoPANDA core module
        print("Detection logic to be implemented...")
        pass

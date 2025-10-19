"""
Version information command
tomopanda version [options]
"""

import argparse
import sys
from typing import Optional

from .base import BaseCommand


class VersionCommand(BaseCommand):
    """Version information command"""
    
    def get_name(self) -> str:
        return "version"
    
    def get_description(self) -> str:
        return "Display TomoPANDA version information"
    
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description="Display TomoPANDA version information and system information"
        )
        
        # Version parameters
        parser.add_argument(
            '--short', '-s',
            action='store_true',
            help='Show only version number'
        )
        
        # Add common arguments
        self.add_common_args(parser)
        
        return parser
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute version command"""
        try:
            if args.short:
                print(self._get_version())
            else:
                self._print_version_info(args.verbose)
            return 0
            
        except Exception as e:
            print(f"Error occurred while getting version information: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _get_version(self) -> str:
        """Get version number"""
        try:
            from tomopanda import __version__
            return __version__
        except ImportError:
            return "0.1.0"
    
    def _print_version_info(self, verbose: bool) -> None:
        """Print version information"""
        print(f"TomoPANDA {self._get_version()}")
        
        if verbose:
            print(f"Python version: {sys.version}")
            print(f"Platform: {sys.platform}")
            
            # 显示依赖包版本
            try:
                import torch
                print(f"PyTorch version: {torch.__version__}")
            except ImportError:
                print("PyTorch: Not installed")
            
            try:
                import numpy
                print(f"NumPy version: {numpy.__version__}")
            except ImportError:
                print("NumPy: Not installed")
            
            try:
                import scipy
                print(f"SciPy version: {scipy.__version__}")
            except ImportError:
                print("SciPy: Not installed")

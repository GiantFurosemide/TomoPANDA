"""
版本信息命令
tomopanda version [options]
"""

import argparse
import sys
from typing import Optional

from .base import BaseCommand


class VersionCommand(BaseCommand):
    """版本信息命令"""
    
    def get_name(self) -> str:
        return "version"
    
    def get_description(self) -> str:
        return "显示TomoPANDA版本信息"
    
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description="显示TomoPANDA的版本信息和系统信息"
        )
        
        # 版本参数
        parser.add_argument(
            '--short', '-s',
            action='store_true',
            help='只显示版本号'
        )
        
        # 添加通用参数
        self.add_common_args(parser)
        
        return parser
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行版本命令"""
        try:
            if args.short:
                print(self._get_version())
            else:
                self._print_version_info(args.verbose)
            return 0
            
        except Exception as e:
            print(f"获取版本信息时发生错误: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _get_version(self) -> str:
        """获取版本号"""
        try:
            from tomopanda import __version__
            return __version__
        except ImportError:
            return "0.1.0"
    
    def _print_version_info(self, verbose: bool) -> None:
        """打印版本信息"""
        print(f"TomoPANDA {self._get_version()}")
        
        if verbose:
            print(f"Python版本: {sys.version}")
            print(f"平台: {sys.platform}")
            
            # 显示依赖包版本
            try:
                import torch
                print(f"PyTorch版本: {torch.__version__}")
            except ImportError:
                print("PyTorch: 未安装")
            
            try:
                import numpy
                print(f"NumPy版本: {numpy.__version__}")
            except ImportError:
                print("NumPy: 未安装")
            
            try:
                import scipy
                print(f"SciPy版本: {scipy.__version__}")
            except ImportError:
                print("SciPy: 未安装")

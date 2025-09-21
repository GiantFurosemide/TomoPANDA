"""
TomoPANDA - CryoET膜蛋白检测工具
基于SE(3)等变变换器的断层扫描数据分析工具
"""

__version__ = "0.1.0"
__author__ = "TomoPANDA Team"
__email__ = "contact@tomopanda.org"

from .cli import main

# Only export main CLI entry point
__all__ = ["main"]

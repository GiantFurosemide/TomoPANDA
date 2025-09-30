"""
TomoPANDA - CryoET膜蛋白检测工具
基于SE(3)等变变换器的断层扫描数据分析工具
"""

__version__ = "0.1.1"
__author__ = "TomoPANDA Team"
__email__ = "contact@tomopanda.org"

from .cli import main

# Re-export selected mesh geodesic helpers for convenience
from .core.mesh_geodesic import (
    create_mesh_geodesic_sampler,
    generate_synthetic_mask,
    run_mesh_geodesic_sampling,
    save_sampling_outputs,
    create_visualization_script,
)

__all__ = [
    "main",
    "create_mesh_geodesic_sampler",
    "generate_synthetic_mask",
    "run_mesh_geodesic_sampling",
    "save_sampling_outputs",
    "create_visualization_script",
]

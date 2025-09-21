"""
SE(3)等变变换器核心实现
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from .irreducible_representations import IrreducibleRepresentations
from .spherical_harmonics import SphericalHarmonics
from .group_convolution import GroupConvolution


class SE3Transformer(nn.Module):
    """
    SE(3)等变变换器主架构
    
    基于SE(3)群等变性的3D变换器，用于CryoET膜蛋白检测
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 6,
        max_degree: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_degree = max_degree
        self.dropout = dropout
        
        # 初始化组件
        self.irreducible_reps = IrreducibleRepresentations(max_degree)
        self.spherical_harmonics = SphericalHarmonics(max_degree)
        
        # 构建网络层
        self._build_layers()
        
    def _build_layers(self):
        """构建网络层"""
        self.layers = nn.ModuleList()
        
        # 输入层
        self.input_layer = GroupConvolution(
            in_type=self.irreducible_reps.get_type(self.input_dim),
            out_type=self.irreducible_reps.get_type(self.hidden_dim),
            max_degree=self.max_degree
        )
        
        # 隐藏层
        for i in range(self.num_layers):
            layer = nn.ModuleDict({
                'conv': GroupConvolution(
                    in_type=self.irreducible_reps.get_type(self.hidden_dim),
                    out_type=self.irreducible_reps.get_type(self.hidden_dim),
                    max_degree=self.max_degree
                ),
                'norm': nn.LayerNorm(self.hidden_dim),
                'dropout': nn.Dropout(self.dropout)
            })
            self.layers.append(layer)
        
        # 输出层
        self.output_layer = GroupConvolution(
            in_type=self.irreducible_reps.get_type(self.hidden_dim),
            out_type=self.irreducible_reps.get_type(self.output_dim),
            max_degree=self.max_degree
        )
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, num_points, input_dim]
            positions: 3D位置 [batch_size, num_points, 3]
            
        Returns:
            输出特征 [batch_size, num_points, output_dim]
        """
        batch_size, num_points, _ = x.shape
        
        # 转换为球谐函数表示
        spherical_coords = self._cartesian_to_spherical(positions)
        harmonics = self.spherical_harmonics(spherical_coords)
        
        # 输入层
        x = self.input_layer(x, harmonics)
        
        # 隐藏层
        for layer in self.layers:
            residual = x
            x = layer['conv'](x, harmonics)
            x = layer['norm'](x)
            x = torch.relu(x)
            x = layer['dropout'](x)
            x = x + residual  # 残差连接
        
        # 输出层
        x = self.output_layer(x, harmonics)
        
        return x
    
    def _cartesian_to_spherical(self, positions: torch.Tensor) -> torch.Tensor:
        """将笛卡尔坐标转换为球坐标"""
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / (r + 1e-8))  # 极角
        phi = torch.atan2(y, x)  # 方位角
        
        return torch.stack([r, theta, phi], dim=-1)
    
    def get_equivariant_features(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """获取等变特征"""
        return self.forward(x, positions)
    
    def apply_se3_transform(self, x: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """应用SE(3)变换"""
        # 应用旋转和平移
        transformed_positions = torch.matmul(positions, rotation.transpose(-1, -2)) + translation
        return self.forward(x, transformed_positions)

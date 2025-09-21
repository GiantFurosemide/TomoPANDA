"""
数学工具函数
包含四元数、旋转矩阵、李代数等数学计算
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional
import math


class MathUtils:
    """
    数学工具类
    
    提供四元数、旋转矩阵、李代数等数学计算功能
    """
    
    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """
        四元数转旋转矩阵
        
        Args:
            quaternion: 四元数 [w, x, y, z] 或 [x, y, z, w]
            
        Returns:
            旋转矩阵 [3, 3]
        """
        if quaternion.shape[-1] == 4:
            if quaternion.shape[0] == 4:  # [w, x, y, z]
                w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            else:  # [x, y, z, w]
                x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        else:
            raise ValueError("四元数必须是4维向量")
        
        # 归一化四元数
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # 计算旋转矩阵
        R = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
        
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - w*z)
        R[0, 2] = 2*(x*z + w*y)
        
        R[1, 0] = 2*(x*y + w*z)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - w*x)
        
        R[2, 0] = 2*(x*z - w*y)
        R[2, 1] = 2*(y*z + w*x)
        R[2, 2] = 1 - 2*(x*x + y*y)
        
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        旋转矩阵转四元数
        
        Args:
            rotation_matrix: 旋转矩阵 [3, 3]
            
        Returns:
            四元数 [w, x, y, z]
        """
        R = rotation_matrix
        
        # 计算四元数分量
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return torch.stack([w, x, y, z])
    
    @staticmethod
    def euler_angles_to_rotation_matrix(angles: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
        """
        欧拉角转旋转矩阵
        
        Args:
            angles: 欧拉角 [alpha, beta, gamma]
            order: 旋转顺序 ('xyz', 'zyx', 'yxz', etc.)
            
        Returns:
            旋转矩阵 [3, 3]
        """
        alpha, beta, gamma = angles[0], angles[1], angles[2]
        
        # 计算旋转矩阵
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(alpha), -torch.sin(alpha)],
            [0, torch.sin(alpha), torch.cos(alpha)]
        ], dtype=angles.dtype, device=angles.device)
        
        Ry = torch.tensor([
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)]
        ], dtype=angles.dtype, device=angles.device)
        
        Rz = torch.tensor([
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1]
        ], dtype=angles.dtype, device=angles.device)
        
        # 根据顺序组合旋转矩阵
        if order == 'xyz':
            return torch.matmul(torch.matmul(Rz, Ry), Rx)
        elif order == 'zyx':
            return torch.matmul(torch.matmul(Rx, Ry), Rz)
        elif order == 'yxz':
            return torch.matmul(torch.matmul(Rz, Rx), Ry)
        else:
            raise ValueError(f"不支持的旋转顺序: {order}")
    
    @staticmethod
    def axis_angle_to_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        轴角表示转旋转矩阵
        
        Args:
            axis: 旋转轴 [x, y, z]
            angle: 旋转角度
            
        Returns:
            旋转矩阵 [3, 3]
        """
        # 归一化旋转轴
        axis = axis / torch.norm(axis)
        
        # 计算旋转矩阵
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        x, y, z = axis[0], axis[1], axis[2]
        
        R = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
        
        R[0, 0] = cos_angle + x*x*(1 - cos_angle)
        R[0, 1] = x*y*(1 - cos_angle) - z*sin_angle
        R[0, 2] = x*z*(1 - cos_angle) + y*sin_angle
        
        R[1, 0] = y*x*(1 - cos_angle) + z*sin_angle
        R[1, 1] = cos_angle + y*y*(1 - cos_angle)
        R[1, 2] = y*z*(1 - cos_angle) - x*sin_angle
        
        R[2, 0] = z*x*(1 - cos_angle) - y*sin_angle
        R[2, 1] = z*y*(1 - cos_angle) + x*sin_angle
        R[2, 2] = cos_angle + z*z*(1 - cos_angle)
        
        return R
    
    @staticmethod
    def se3_to_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """
        SE(3)变换转齐次变换矩阵
        
        Args:
            rotation: 旋转矩阵 [3, 3]
            translation: 平移向量 [3]
            
        Returns:
            齐次变换矩阵 [4, 4]
        """
        T = torch.zeros(4, 4, dtype=rotation.dtype, device=rotation.device)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        T[3, 3] = 1.0
        
        return T
    
    @staticmethod
    def matrix_to_se3(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        齐次变换矩阵转SE(3)变换
        
        Args:
            matrix: 齐次变换矩阵 [4, 4]
            
        Returns:
            旋转矩阵和平移向量
        """
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        
        return rotation, translation
    
    @staticmethod
    def compose_se3(transform1: torch.Tensor, transform2: torch.Tensor) -> torch.Tensor:
        """
        组合两个SE(3)变换
        
        Args:
            transform1: 第一个变换 [4, 4]
            transform2: 第二个变换 [4, 4]
            
        Returns:
            组合后的变换 [4, 4]
        """
        return torch.matmul(transform2, transform1)
    
    @staticmethod
    def inverse_se3(transform: torch.Tensor) -> torch.Tensor:
        """
        计算SE(3)变换的逆
        
        Args:
            transform: SE(3)变换矩阵 [4, 4]
            
        Returns:
            逆变换矩阵 [4, 4]
        """
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        
        inv_rotation = rotation.transpose(-1, -2)
        inv_translation = -torch.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(-1)
        
        inv_transform = torch.zeros_like(transform)
        inv_transform[:3, :3] = inv_rotation
        inv_transform[:3, 3] = inv_translation
        inv_transform[3, 3] = 1.0
        
        return inv_transform

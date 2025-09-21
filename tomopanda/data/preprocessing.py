"""
数据预处理模块
包含去噪、归一化、增强等预处理功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from scipy import ndimage
from skimage import filters, restoration


class DataPreprocessor:
    """
    数据预处理器
    
    提供断层扫描数据的预处理功能，包括去噪、归一化、增强等
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def preprocess(
        self, 
        data: torch.Tensor, 
        steps: list = None,
        **kwargs
    ) -> torch.Tensor:
        """
        预处理数据
        
        Args:
            data: 输入数据 [depth, height, width] 或 [batch_size, depth, height, width]
            steps: 预处理步骤列表
            **kwargs: 额外参数
            
        Returns:
            预处理后的数据
        """
        if steps is None:
            steps = ['normalize', 'denoise']
        
        processed_data = data.clone()
        
        for step in steps:
            if step == 'normalize':
                processed_data = self.normalize(processed_data, **kwargs)
            elif step == 'denoise':
                processed_data = self.denoise(processed_data, **kwargs)
            elif step == 'enhance':
                processed_data = self.enhance(processed_data, **kwargs)
            elif step == 'crop':
                processed_data = self.crop(processed_data, **kwargs)
            elif step == 'resize':
                processed_data = self.resize(processed_data, **kwargs)
            else:
                print(f"警告: 未知的预处理步骤: {step}")
        
        return processed_data
    
    def normalize(
        self, 
        data: torch.Tensor, 
        method: str = 'zscore',
        **kwargs
    ) -> torch.Tensor:
        """
        数据归一化
        
        Args:
            data: 输入数据
            method: 归一化方法 ('zscore', 'minmax', 'robust')
            **kwargs: 额外参数
            
        Returns:
            归一化后的数据
        """
        if method == 'zscore':
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'robust':
            median = data.median()
            mad = torch.median(torch.abs(data - median))
            return (data - median) / (mad + 1e-8)
        
        else:
            raise ValueError(f"未知的归一化方法: {method}")
    
    def denoise(
        self, 
        data: torch.Tensor, 
        method: str = 'gaussian',
        **kwargs
    ) -> torch.Tensor:
        """
        数据去噪
        
        Args:
            data: 输入数据
            method: 去噪方法 ('gaussian', 'bilateral', 'non_local_means')
            **kwargs: 额外参数
            
        Returns:
            去噪后的数据
        """
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return self._gaussian_filter(data, sigma)
        
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.1)
            sigma_spatial = kwargs.get('sigma_spatial', 1.0)
            return self._bilateral_filter(data, sigma_color, sigma_spatial)
        
        elif method == 'non_local_means':
            h = kwargs.get('h', 0.1)
            return self._non_local_means(data, h)
        
        else:
            raise ValueError(f"未知的去噪方法: {method}")
    
    def enhance(
        self, 
        data: torch.Tensor, 
        method: str = 'clahe',
        **kwargs
    ) -> torch.Tensor:
        """
        数据增强
        
        Args:
            data: 输入数据
            method: 增强方法 ('clahe', 'gamma', 'histogram_equalization')
            **kwargs: 额外参数
            
        Returns:
            增强后的数据
        """
        if method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 0.01)
            return self._clahe(data, clip_limit)
        
        elif method == 'gamma':
            gamma = kwargs.get('gamma', 1.0)
            return self._gamma_correction(data, gamma)
        
        elif method == 'histogram_equalization':
            return self._histogram_equalization(data)
        
        else:
            raise ValueError(f"未知的增强方法: {method}")
    
    def crop(
        self, 
        data: torch.Tensor, 
        crop_size: Union[int, Tuple[int, int, int]],
        center: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        数据裁剪
        
        Args:
            data: 输入数据
            crop_size: 裁剪尺寸
            center: 裁剪中心点
            
        Returns:
            裁剪后的数据
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size, crop_size)
        
        if center is None:
            center = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)
        
        start = [c - s // 2 for c, s in zip(center, crop_size)]
        end = [s + c for s, c in zip(start, crop_size)]
        
        return data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    def resize(
        self, 
        data: torch.Tensor, 
        target_size: Union[int, Tuple[int, int, int]],
        method: str = 'trilinear'
    ) -> torch.Tensor:
        """
        数据缩放
        
        Args:
            data: 输入数据
            target_size: 目标尺寸
            method: 缩放方法
            
        Returns:
            缩放后的数据
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        
        # 添加batch维度
        if data.dim() == 3:
            data = data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 使用PyTorch的插值函数
        resized = F.interpolate(
            data.unsqueeze(0),  # 添加channel维度
            size=target_size,
            mode=method,
            align_corners=False
        ).squeeze(0).squeeze(0)  # 移除channel维度
        
        if squeeze_output:
            resized = resized.squeeze(0)
        
        return resized
    
    def _gaussian_filter(self, data: torch.Tensor, sigma: float) -> torch.Tensor:
        """高斯滤波"""
        # 转换为numpy进行滤波
        data_np = data.cpu().numpy()
        filtered = ndimage.gaussian_filter(data_np, sigma=sigma)
        return torch.from_numpy(filtered).to(self.device)
    
    def _bilateral_filter(self, data: torch.Tensor, sigma_color: float, sigma_spatial: float) -> torch.Tensor:
        """双边滤波"""
        # 转换为numpy进行滤波
        data_np = data.cpu().numpy()
        filtered = filters.bilateral(data_np, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        return torch.from_numpy(filtered).to(self.device)
    
    def _non_local_means(self, data: torch.Tensor, h: float) -> torch.Tensor:
        """非局部均值去噪"""
        # 转换为numpy进行滤波
        data_np = data.cpu().numpy()
        filtered = restoration.denoise_nl_means(data_np, h=h)
        return torch.from_numpy(filtered).to(self.device)
    
    def _clahe(self, data: torch.Tensor, clip_limit: float) -> torch.Tensor:
        """对比度限制自适应直方图均衡化"""
        # 转换为numpy进行处理
        data_np = data.cpu().numpy()
        # 这里需要实现CLAHE算法
        # 实际实现需要复杂的图像处理算法
        return torch.from_numpy(data_np).to(self.device)
    
    def _gamma_correction(self, data: torch.Tensor, gamma: float) -> torch.Tensor:
        """伽马校正"""
        return torch.pow(data, gamma)
    
    def _histogram_equalization(self, data: torch.Tensor) -> torch.Tensor:
        """直方图均衡化"""
        # 转换为numpy进行处理
        data_np = data.cpu().numpy()
        # 这里需要实现直方图均衡化算法
        # 实际实现需要复杂的图像处理算法
        return torch.from_numpy(data_np).to(self.device)

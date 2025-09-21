"""
断层扫描数据加载器
支持多种格式的CryoET数据加载
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import mrcfile
import h5py


class TomogramLoader:
    """
    断层扫描数据加载器
    
    支持多种格式的CryoET数据加载，包括MRC、HDF5、Zarr等
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.supported_formats = ['.mrc', '.mrcs', '.h5', '.hdf5', '.zarr']
    
    def load(self, file_path: Union[str, Path], **kwargs) -> torch.Tensor:
        """
        加载断层扫描数据
        
        Args:
            file_path: 数据文件路径
            **kwargs: 额外参数
            
        Returns:
            加载的数据张量 [depth, height, width]
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.mrc', '.mrcs']:
            return self._load_mrc(file_path, **kwargs)
        elif suffix in ['.h5', '.hdf5']:
            return self._load_hdf5(file_path, **kwargs)
        elif suffix == '.zarr':
            return self._load_zarr(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _load_mrc(self, file_path: Path, **kwargs) -> torch.Tensor:
        """加载MRC格式数据"""
        try:
            with mrcfile.open(file_path, mode='r') as mrc:
                data = mrc.data
                # 转换为PyTorch张量
                tensor = torch.from_numpy(data.astype(np.float32))
                return tensor.to(self.device)
        except Exception as e:
            raise RuntimeError(f"加载MRC文件失败: {e}")
    
    def _load_hdf5(self, file_path: Path, dataset_path: str = '/data', **kwargs) -> torch.Tensor:
        """加载HDF5格式数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                data = f[dataset_path][:]
                tensor = torch.from_numpy(data.astype(np.float32))
                return tensor.to(self.device)
        except Exception as e:
            raise RuntimeError(f"加载HDF5文件失败: {e}")
    
    def _load_zarr(self, file_path: Path, **kwargs) -> torch.Tensor:
        """加载Zarr格式数据"""
        try:
            import zarr
            z = zarr.open(file_path, mode='r')
            data = z[:]
            tensor = torch.from_numpy(data.astype(np.float32))
            return tensor.to(self.device)
        except Exception as e:
            raise RuntimeError(f"加载Zarr文件失败: {e}")
    
    def load_batch(self, file_paths: list, **kwargs) -> torch.Tensor:
        """
        批量加载数据
        
        Args:
            file_paths: 文件路径列表
            **kwargs: 额外参数
            
        Returns:
            批量数据张量 [batch_size, depth, height, width]
        """
        batch_data = []
        
        for file_path in file_paths:
            data = self.load(file_path, **kwargs)
            batch_data.append(data)
        
        return torch.stack(batch_data, dim=0)
    
    def get_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取数据文件信息
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据信息字典
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in ['.mrc', '.mrcs']:
            return self._get_mrc_info(file_path)
        elif suffix in ['.h5', '.hdf5']:
            return self._get_hdf5_info(file_path)
        else:
            return {"error": f"不支持的文件格式: {suffix}"}
    
    def _get_mrc_info(self, file_path: Path) -> Dict[str, Any]:
        """获取MRC文件信息"""
        try:
            with mrcfile.open(file_path, mode='r') as mrc:
                return {
                    "shape": mrc.data.shape,
                    "dtype": str(mrc.data.dtype),
                    "voxel_size": mrc.voxel_size,
                    "header": dict(mrc.header)
                }
        except Exception as e:
            return {"error": f"获取MRC信息失败: {e}"}
    
    def _get_hdf5_info(self, file_path: Path) -> Dict[str, Any]:
        """获取HDF5文件信息"""
        try:
            with h5py.File(file_path, 'r') as f:
                return {
                    "keys": list(f.keys()),
                    "shape": f['/data'].shape if '/data' in f else "unknown",
                    "dtype": str(f['/data'].dtype) if '/data' in f else "unknown"
                }
        except Exception as e:
            return {"error": f"获取HDF5信息失败: {e}"}

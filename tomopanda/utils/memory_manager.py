"""
内存管理工具
包含内存监控、缓存管理、垃圾回收等功能
"""

import torch
import gc
import psutil
import os
from typing import Dict, Any, Optional, List
import threading
import time
from pathlib import Path


class MemoryManager:
    """
    内存管理器
    
    提供内存监控、缓存管理、垃圾回收等功能
    """
    
    def __init__(self, max_memory_gb: float = 8.0, cache_dir: str = "./cache"):
        self.max_memory_gb = max_memory_gb
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 内存监控
        self.memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
        # 缓存管理
        self.cache = {}
        self.cache_size = 0
        self.max_cache_size = max_memory_gb * 0.5  # 使用50%的内存作为缓存
    
    def start_monitoring(self, interval: float = 1.0):
        """开始内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_memory(self, interval: float):
        """监控内存使用情况"""
        while self.monitoring:
            memory_info = self.get_memory_info()
            self.memory_usage.append(memory_info)
            
            # 检查内存使用是否过高
            if memory_info['total_usage_gb'] > self.max_memory_gb:
                self._handle_memory_pressure()
            
            time.sleep(interval)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        
        # PyTorch内存信息
        torch_memory = {}
        if torch.cuda.is_available():
            torch_memory = {
                'cuda_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cuda_cached': torch.cuda.memory_reserved() / 1024**3,  # GB
                'cuda_max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        
        return {
            'system_total': system_memory.total / 1024**3,  # GB
            'system_available': system_memory.available / 1024**3,  # GB
            'system_used': system_memory.used / 1024**3,  # GB
            'system_percent': system_memory.percent,
            'total_usage_gb': system_memory.used / 1024**3,
            'torch_memory': torch_memory,
            'cache_size_gb': self.cache_size,
            'timestamp': time.time()
        }
    
    def _handle_memory_pressure(self):
        """处理内存压力"""
        print("警告: 内存使用过高，开始清理...")
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理Python缓存
        self.clear_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        print("内存清理完成")
    
    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
        self.cache_size = 0
        
        # 清理缓存目录
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
    
    def cache_tensor(self, key: str, tensor: torch.Tensor, persistent: bool = False) -> str:
        """
        缓存张量
        
        Args:
            key: 缓存键
            tensor: 要缓存的张量
            persistent: 是否持久化到磁盘
            
        Returns:
            缓存路径
        """
        tensor_size_gb = tensor.element_size() * tensor.nelement() / 1024**3
        
        # 检查缓存大小
        if self.cache_size + tensor_size_gb > self.max_cache_size:
            self._evict_cache()
        
        if persistent:
            # 持久化到磁盘
            cache_path = self.cache_dir / f"{key}.pt"
            torch.save(tensor, cache_path)
            return str(cache_path)
        else:
            # 内存缓存
            self.cache[key] = tensor
            self.cache_size += tensor_size_gb
            return key
    
    def load_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        加载缓存的张量
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的张量或None
        """
        if key in self.cache:
            return self.cache[key]
        
        # 尝试从磁盘加载
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        
        return None
    
    def _evict_cache(self):
        """清理缓存以释放内存"""
        if not self.cache:
            return
        
        # 简单的LRU策略：删除最旧的缓存
        oldest_key = next(iter(self.cache))
        tensor = self.cache.pop(oldest_key)
        tensor_size_gb = tensor.element_size() * tensor.nelement() / 1024**3
        self.cache_size -= tensor_size_gb
        
        print(f"清理缓存: {oldest_key}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存使用报告"""
        current_info = self.get_memory_info()
        
        # 计算统计信息
        if self.memory_usage:
            usage_gb = [info['total_usage_gb'] for info in self.memory_usage]
            stats = {
                'min_usage_gb': min(usage_gb),
                'max_usage_gb': max(usage_gb),
                'avg_usage_gb': sum(usage_gb) / len(usage_gb),
                'current_usage_gb': current_info['total_usage_gb']
            }
        else:
            stats = {
                'min_usage_gb': current_info['total_usage_gb'],
                'max_usage_gb': current_info['total_usage_gb'],
                'avg_usage_gb': current_info['total_usage_gb'],
                'current_usage_gb': current_info['total_usage_gb']
            }
        
        return {
            'current_info': current_info,
            'statistics': stats,
            'cache_info': {
                'cache_size_gb': self.cache_size,
                'max_cache_size_gb': self.max_cache_size,
                'num_cached_items': len(self.cache)
            }
        }
    
    def optimize_memory(self):
        """优化内存使用"""
        print("开始内存优化...")
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理Python缓存
        self.clear_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取优化后的内存信息
        memory_info = self.get_memory_info()
        print(f"内存优化完成，当前使用: {memory_info['total_usage_gb']:.2f} GB")
    
    def set_memory_limit(self, limit_gb: float):
        """设置内存限制"""
        self.max_memory_gb = limit_gb
        self.max_cache_size = limit_gb * 0.5
        print(f"内存限制设置为: {limit_gb} GB")
    
    def get_recommendations(self) -> List[str]:
        """获取内存优化建议"""
        recommendations = []
        memory_info = self.get_memory_info()
        
        if memory_info['total_usage_gb'] > self.max_memory_gb * 0.8:
            recommendations.append("内存使用过高，建议减少批处理大小")
        
        if memory_info['torch_memory'].get('cuda_allocated', 0) > 0:
            recommendations.append("GPU内存使用中，考虑使用CPU处理")
        
        if self.cache_size > self.max_cache_size * 0.8:
            recommendations.append("缓存使用过高，建议清理缓存")
        
        if memory_info['system_percent'] > 90:
            recommendations.append("系统内存使用过高，建议关闭其他程序")
        
        return recommendations

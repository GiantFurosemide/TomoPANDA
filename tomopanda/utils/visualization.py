"""
可视化工具
包含断层扫描数据、粒子检测结果的可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, Optional, List, Tuple, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class Visualizer:
    """
    可视化工具类
    
    提供断层扫描数据、粒子检测结果的可视化功能
    """
    
    def __init__(self, backend: str = 'matplotlib'):
        self.backend = backend
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    def visualize_tomogram(
        self, 
        data: torch.Tensor, 
        slice_idx: Optional[int] = None,
        projection: str = 'xy',
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        可视化断层扫描数据
        
        Args:
            data: 断层扫描数据 [depth, height, width]
            slice_idx: 切片索引
            projection: 投影方向 ('xy', 'xz', 'yz', 'max', 'mean')
            save_path: 保存路径
            **kwargs: 额外参数
        """
        if self.backend == 'matplotlib':
            self._visualize_tomogram_matplotlib(data, slice_idx, projection, save_path, **kwargs)
        elif self.backend == 'plotly':
            self._visualize_tomogram_plotly(data, slice_idx, projection, save_path, **kwargs)
        else:
            raise ValueError(f"不支持的后端: {self.backend}")
    
    def _visualize_tomogram_matplotlib(
        self, 
        data: torch.Tensor, 
        slice_idx: Optional[int],
        projection: str,
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用matplotlib可视化断层扫描数据"""
        data_np = data.cpu().numpy()
        
        if projection == 'xy':
            if slice_idx is not None:
                image = data_np[slice_idx]
            else:
                image = data_np[data_np.shape[0] // 2]
        elif projection == 'xz':
            if slice_idx is not None:
                image = data_np[:, slice_idx, :]
            else:
                image = data_np[:, data_np.shape[1] // 2, :]
        elif projection == 'yz':
            if slice_idx is not None:
                image = data_np[:, :, slice_idx]
            else:
                image = data_np[:, :, data_np.shape[2] // 2]
        elif projection == 'max':
            image = np.max(data_np, axis=0)
        elif projection == 'mean':
            image = np.mean(data_np, axis=0)
        else:
            raise ValueError(f"不支持的投影方向: {projection}")
        
        plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        plt.imshow(image, cmap=kwargs.get('colormap', 'viridis'))
        plt.colorbar()
        plt.title(f'断层扫描数据 - {projection}投影')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        else:
            plt.show()
    
    def _visualize_tomogram_plotly(
        self, 
        data: torch.Tensor, 
        slice_idx: Optional[int],
        projection: str,
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用plotly可视化断层扫描数据"""
        data_np = data.cpu().numpy()
        
        if projection == 'xy':
            if slice_idx is not None:
                image = data_np[slice_idx]
            else:
                image = data_np[data_np.shape[0] // 2]
        elif projection == 'xz':
            if slice_idx is not None:
                image = data_np[:, slice_idx, :]
            else:
                image = data_np[:, data_np.shape[1] // 2, :]
        elif projection == 'yz':
            if slice_idx is not None:
                image = data_np[:, :, slice_idx]
            else:
                image = data_np[:, :, data_np.shape[2] // 2]
        elif projection == 'max':
            image = np.max(data_np, axis=0)
        elif projection == 'mean':
            image = np.mean(data_np, axis=0)
        else:
            raise ValueError(f"不支持的投影方向: {projection}")
        
        fig = go.Figure(data=go.Heatmap(z=image, colorscale='Viridis'))
        fig.update_layout(
            title=f'断层扫描数据 - {projection}投影',
            xaxis_title='X',
            yaxis_title='Y'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def visualize_particles(
        self, 
        particles: torch.Tensor, 
        positions: torch.Tensor,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        可视化粒子检测结果
        
        Args:
            particles: 粒子特征 [N, feature_dim]
            positions: 粒子位置 [N, 3]
            save_path: 保存路径
            **kwargs: 额外参数
        """
        if self.backend == 'matplotlib':
            self._visualize_particles_matplotlib(particles, positions, save_path, **kwargs)
        elif self.backend == 'plotly':
            self._visualize_particles_plotly(particles, positions, save_path, **kwargs)
        else:
            raise ValueError(f"不支持的后端: {self.backend}")
    
    def _visualize_particles_matplotlib(
        self, 
        particles: torch.Tensor, 
        positions: torch.Tensor,
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用matplotlib可视化粒子"""
        positions_np = positions.cpu().numpy()
        
        fig = plt.figure(figsize=kwargs.get('figsize', (12, 8)))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制粒子
        scatter = ax.scatter(
            positions_np[:, 0], 
            positions_np[:, 1], 
            positions_np[:, 2],
            c=particles.cpu().numpy()[:, 0] if particles.shape[1] > 0 else 'blue',
            cmap=kwargs.get('colormap', 'viridis'),
            s=kwargs.get('size', 50),
            alpha=kwargs.get('alpha', 0.8)
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('粒子检测结果')
        
        if particles.shape[1] > 0:
            plt.colorbar(scatter, ax=ax, label='粒子强度')
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        else:
            plt.show()
    
    def _visualize_particles_plotly(
        self, 
        particles: torch.Tensor, 
        positions: torch.Tensor,
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用plotly可视化粒子"""
        positions_np = positions.cpu().numpy()
        
        fig = go.Figure(data=go.Scatter3d(
            x=positions_np[:, 0],
            y=positions_np[:, 1],
            z=positions_np[:, 2],
            mode='markers',
            marker=dict(
                size=kwargs.get('size', 5),
                color=particles.cpu().numpy()[:, 0] if particles.shape[1] > 0 else 'blue',
                colorscale='Viridis',
                opacity=kwargs.get('alpha', 0.8)
            ),
            text=[f'粒子 {i}' for i in range(len(positions_np))],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x}<br>' +
                         'Y: %{y}<br>' +
                         'Z: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='粒子检测结果',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def visualize_trajectory(
        self, 
        trajectory: torch.Tensor, 
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        可视化轨迹
        
        Args:
            trajectory: 轨迹数据 [T, 3]
            save_path: 保存路径
            **kwargs: 额外参数
        """
        if self.backend == 'matplotlib':
            self._visualize_trajectory_matplotlib(trajectory, save_path, **kwargs)
        elif self.backend == 'plotly':
            self._visualize_trajectory_plotly(trajectory, save_path, **kwargs)
        else:
            raise ValueError(f"不支持的后端: {self.backend}")
    
    def _visualize_trajectory_matplotlib(
        self, 
        trajectory: torch.Tensor, 
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用matplotlib可视化轨迹"""
        trajectory_np = trajectory.cpu().numpy()
        
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(
            trajectory_np[:, 0], 
            trajectory_np[:, 1], 
            trajectory_np[:, 2],
            'b-', 
            linewidth=kwargs.get('linewidth', 2),
            alpha=kwargs.get('alpha', 0.8)
        )
        
        # 标记起点和终点
        ax.scatter(
            trajectory_np[0, 0], 
            trajectory_np[0, 1], 
            trajectory_np[0, 2],
            c='green', 
            s=100, 
            label='起点'
        )
        ax.scatter(
            trajectory_np[-1, 0], 
            trajectory_np[-1, 1], 
            trajectory_np[-1, 2],
            c='red', 
            s=100, 
            label='终点'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('粒子轨迹')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        else:
            plt.show()
    
    def _visualize_trajectory_plotly(
        self, 
        trajectory: torch.Tensor, 
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用plotly可视化轨迹"""
        trajectory_np = trajectory.cpu().numpy()
        
        fig = go.Figure()
        
        # 绘制轨迹
        fig.add_trace(go.Scatter3d(
            x=trajectory_np[:, 0],
            y=trajectory_np[:, 1],
            z=trajectory_np[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=kwargs.get('linewidth', 2)),
            marker=dict(size=3),
            name='轨迹'
        ))
        
        # 标记起点和终点
        fig.add_trace(go.Scatter3d(
            x=[trajectory_np[0, 0]],
            y=[trajectory_np[0, 1]],
            z=[trajectory_np[0, 2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='起点'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory_np[-1, 0]],
            y=[trajectory_np[-1, 1]],
            z=[trajectory_np[-1, 2]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='终点'
        ))
        
        fig.update_layout(
            title='粒子轨迹',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def visualize_statistics(
        self, 
        data: Dict[str, Any], 
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        可视化统计信息
        
        Args:
            data: 统计数据字典
            save_path: 保存路径
            **kwargs: 额外参数
        """
        if self.backend == 'matplotlib':
            self._visualize_statistics_matplotlib(data, save_path, **kwargs)
        elif self.backend == 'plotly':
            self._visualize_statistics_plotly(data, save_path, **kwargs)
        else:
            raise ValueError(f"不支持的后端: {self.backend}")
    
    def _visualize_statistics_matplotlib(
        self, 
        data: Dict[str, Any], 
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用matplotlib可视化统计信息"""
        fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (12, 10)))
        axes = axes.flatten()
        
        # 绘制直方图
        if 'histogram' in data:
            axes[0].hist(data['histogram'], bins=50, alpha=0.7)
            axes[0].set_title('数据分布')
            axes[0].set_xlabel('值')
            axes[0].set_ylabel('频次')
        
        # 绘制箱线图
        if 'boxplot' in data:
            axes[1].boxplot(data['boxplot'])
            axes[1].set_title('数据分布')
            axes[1].set_ylabel('值')
        
        # 绘制散点图
        if 'scatter' in data:
            x, y = data['scatter']
            axes[2].scatter(x, y, alpha=0.6)
            axes[2].set_title('散点图')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
        
        # 绘制时间序列
        if 'timeseries' in data:
            t, values = data['timeseries']
            axes[3].plot(t, values)
            axes[3].set_title('时间序列')
            axes[3].set_xlabel('时间')
            axes[3].set_ylabel('值')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        else:
            plt.show()
    
    def _visualize_statistics_plotly(
        self, 
        data: Dict[str, Any], 
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """使用plotly可视化统计信息"""
        fig = go.Figure()
        
        # 绘制直方图
        if 'histogram' in data:
            fig.add_trace(go.Histogram(x=data['histogram'], name='数据分布'))
        
        # 绘制箱线图
        if 'boxplot' in data:
            fig.add_trace(go.Box(y=data['boxplot'], name='数据分布'))
        
        # 绘制散点图
        if 'scatter' in data:
            x, y = data['scatter']
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='散点图'))
        
        # 绘制时间序列
        if 'timeseries' in data:
            t, values = data['timeseries']
            fig.add_trace(go.Scatter(x=t, y=values, mode='lines', name='时间序列'))
        
        fig.update_layout(title='统计信息')
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

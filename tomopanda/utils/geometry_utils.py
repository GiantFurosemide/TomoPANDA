"""
几何工具函数
包含3D几何计算、距离计算、角度计算等
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, List
import math


class GeometryUtils:
    """
    几何工具类
    
    提供3D几何计算、距离计算、角度计算等功能
    """
    
    @staticmethod
    def distance_point_to_plane(
        point: torch.Tensor, 
        plane_point: torch.Tensor, 
        plane_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算点到平面的距离
        
        Args:
            point: 点坐标 [3]
            plane_point: 平面上一点 [3]
            plane_normal: 平面法向量 [3]
            
        Returns:
            距离
        """
        # 归一化法向量
        plane_normal = plane_normal / torch.norm(plane_normal)
        
        # 计算距离
        distance = torch.abs(torch.dot(point - plane_point, plane_normal))
        
        return distance
    
    @staticmethod
    def distance_point_to_line(
        point: torch.Tensor, 
        line_point: torch.Tensor, 
        line_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        计算点到直线的距离
        
        Args:
            point: 点坐标 [3]
            line_point: 直线上一点 [3]
            line_direction: 直线方向向量 [3]
            
        Returns:
            距离
        """
        # 归一化方向向量
        line_direction = line_direction / torch.norm(line_direction)
        
        # 计算距离
        cross_product = torch.cross(point - line_point, line_direction)
        distance = torch.norm(cross_product)
        
        return distance
    
    @staticmethod
    def angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        计算两个向量之间的角度
        
        Args:
            v1: 第一个向量 [3]
            v2: 第二个向量 [3]
            
        Returns:
            角度（弧度）
        """
        # 归一化向量
        v1_norm = v1 / torch.norm(v1)
        v2_norm = v2 / torch.norm(v2)
        
        # 计算角度
        dot_product = torch.dot(v1_norm, v2_norm)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # 避免数值误差
        angle = torch.acos(dot_product)
        
        return angle
    
    @staticmethod
    def dihedral_angle(
        p1: torch.Tensor, 
        p2: torch.Tensor, 
        p3: torch.Tensor, 
        p4: torch.Tensor
    ) -> torch.Tensor:
        """
        计算二面角
        
        Args:
            p1, p2, p3, p4: 四个点的坐标 [3]
            
        Returns:
            二面角（弧度）
        """
        # 计算向量
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # 计算法向量
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        
        # 归一化法向量
        n1 = n1 / torch.norm(n1)
        n2 = n2 / torch.norm(n2)
        
        # 计算二面角
        dot_product = torch.dot(n1, n2)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle = torch.acos(dot_product)
        
        # 确定符号
        sign = torch.sign(torch.dot(torch.cross(n1, n2), v2))
        angle = angle * sign
        
        return angle
    
    @staticmethod
    def centroid(points: torch.Tensor) -> torch.Tensor:
        """
        计算点集的质心
        
        Args:
            points: 点集 [N, 3]
            
        Returns:
            质心坐标 [3]
        """
        return torch.mean(points, dim=0)
    
    @staticmethod
    def bounding_box(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算点集的边界框
        
        Args:
            points: 点集 [N, 3]
            
        Returns:
            最小点和最大点
        """
        min_point = torch.min(points, dim=0)[0]
        max_point = torch.max(points, dim=0)[0]
        
        return min_point, max_point
    
    @staticmethod
    def convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
        """
        计算2D点集的凸包
        
        Args:
            points: 2D点集 [N, 2]
            
        Returns:
            凸包顶点索引
        """
        # 转换为numpy进行凸包计算
        points_np = points.cpu().numpy()
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_np)
            return torch.from_numpy(hull.vertices).to(points.device)
        except ImportError:
            # 如果没有scipy，使用简单的Graham扫描算法
            return GeometryUtils._graham_scan(points)
    
    @staticmethod
    def _graham_scan(points: torch.Tensor) -> torch.Tensor:
        """Graham扫描算法计算凸包"""
        # 找到最下面的点
        min_y_idx = torch.argmin(points[:, 1])
        min_y_point = points[min_y_idx]
        
        # 按极角排序
        angles = torch.atan2(points[:, 1] - min_y_point[1], points[:, 0] - min_y_point[0])
        sorted_indices = torch.argsort(angles)
        
        # 构建凸包
        hull = [min_y_idx]
        
        for i in range(1, len(sorted_indices)):
            while len(hull) > 1:
                p1 = points[hull[-2]]
                p2 = points[hull[-1]]
                p3 = points[sorted_indices[i]]
                
                # 计算叉积
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                
                if cross <= 0:
                    hull.pop()
                else:
                    break
            
            hull.append(sorted_indices[i].item())
        
        return torch.tensor(hull, dtype=torch.long, device=points.device)
    
    @staticmethod
    def point_in_polygon(point: torch.Tensor, polygon: torch.Tensor) -> bool:
        """
        判断点是否在多边形内
        
        Args:
            point: 点坐标 [2]
            polygon: 多边形顶点 [N, 2]
            
        Returns:
            是否在多边形内
        """
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0, 0], polygon[0, 1]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n, 0], polygon[i % n, 1]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def triangle_area(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """
        计算三角形面积
        
        Args:
            p1, p2, p3: 三角形顶点 [3]
            
        Returns:
            面积
        """
        # 使用叉积计算面积
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = torch.cross(v1, v2)
        area = 0.5 * torch.norm(cross_product)
        
        return area
    
    @staticmethod
    def tetrahedron_volume(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """
        计算四面体体积
        
        Args:
            p1, p2, p3, p4: 四面体顶点 [3]
            
        Returns:
            体积
        """
        # 使用标量三重积计算体积
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1
        
        volume = torch.abs(torch.dot(v1, torch.cross(v2, v3))) / 6.0
        
        return volume

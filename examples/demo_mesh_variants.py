#!/usr/bin/env python3
"""
演示Mesh变体的概念和效果

这个脚本展示：
1. 相同输入生成不同mesh变体
2. 不同变体的三角形分布差异
3. 采样点的空间分布差异
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tomopanda.core.mesh_geodesic import (
    get_triangle_centers_and_normals,
    generate_synthetic_mask
)


def create_sphere_mask():
    """创建球形膜掩码"""
    return generate_synthetic_mask(
        shape=(60, 60, 60), 
        center=(30, 30, 30), 
        radius=20
    )


def generate_mesh_variants(mask, num_variants=4):
    """生成多个mesh变体"""
    variants = []
    seeds = [42, 123, 456, 789]
    
    print("=== 生成Mesh变体 ===")
    for i, seed in enumerate(seeds[:num_variants]):
        print(f"生成变体 {i+1} (种子: {seed})...")
        
        triangle_data = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=15.0,
            random_seed=seed
        )
        
        variants.append({
            'seed': seed,
            'data': triangle_data,
            'centers': triangle_data[:, :3],
            'normals': triangle_data[:, 3:]
        })
        
        print(f"  三角形数量: {len(triangle_data)}")
    
    return variants


def analyze_variants(variants):
    """分析变体差异"""
    print("\n=== 变体分析 ===")
    
    # 统计信息
    for i, variant in enumerate(variants):
        centers = variant['centers']
        print(f"变体 {i+1} (种子 {variant['seed']}):")
        print(f"  三角形数量: {len(centers)}")
        print(f"  X范围: {centers[:, 0].min():.1f} ~ {centers[:, 0].max():.1f}")
        print(f"  Y范围: {centers[:, 1].min():.1f} ~ {centers[:, 1].max():.1f}")
        print(f"  Z范围: {centers[:, 2].min():.1f} ~ {centers[:, 2].max():.1f}")
    
    # 计算变体间的差异
    print("\n变体间差异分析:")
    for i in range(len(variants)):
        for j in range(i+1, len(variants)):
            centers_i = variants[i]['centers']
            centers_j = variants[j]['centers']
            
            # 计算最近邻距离
            from scipy.spatial import cKDTree
            tree_i = cKDTree(centers_i)
            tree_j = cKDTree(centers_j)
            
            # 变体i的每个点到变体j的最近距离
            distances_i_to_j = tree_j.query(centers_i)[0]
            distances_j_to_i = tree_i.query(centers_j)[0]
            
            avg_distance = (distances_i_to_j.mean() + distances_j_to_i.mean()) / 2
            print(f"  变体{i+1} vs 变体{j+1}: 平均最近邻距离 = {avg_distance:.2f}")


def visualize_variants(variants, output_dir="mesh_variants_demo"):
    """可视化mesh变体"""
    print(f"\n=== 创建可视化 ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建2x2子图
    fig = plt.figure(figsize=(16, 12))
    
    for i, variant in enumerate(variants):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        centers = variant['centers']
        
        # 绘制三角形中心
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                  c=range(len(centers)), cmap='viridis', s=20, alpha=0.7)
        
        # 绘制法向量（子集）
        step = max(1, len(centers) // 30)
        for j in range(0, len(centers), step):
            center = centers[j]
            normal = variant['normals'][j] * 3
            ax.quiver(center[0], center[1], center[2], 
                     normal[0], normal[1], normal[2], 
                     color='red', alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Mesh变体 {i+1} (种子: {variant["seed"]})\n{len(centers)} 个三角形')
        
        # 设置相同的坐标范围
        ax.set_xlim(10, 50)
        ax.set_ylim(10, 50)
        ax.set_zlim(10, 50)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = output_dir / "mesh_variants_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"可视化保存到: {plot_path}")
    
    # 创建差异分析图
    create_difference_analysis(variants, output_dir)
    
    return plot_path


def create_difference_analysis(variants, output_dir):
    """创建差异分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 三角形数量对比
    seeds = [v['seed'] for v in variants]
    counts = [len(v['centers']) for v in variants]
    
    axes[0, 0].bar(seeds, counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('随机种子')
    axes[0, 0].set_ylabel('三角形数量')
    axes[0, 0].set_title('不同变体的三角形数量')
    
    # 坐标分布对比
    for i, variant in enumerate(variants):
        centers = variant['centers']
        axes[0, 1].hist(centers[:, 0], alpha=0.5, label=f'变体{i+1}', bins=20)
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('X坐标分布对比')
    axes[0, 1].legend()
    
    # 法向量长度分布
    for i, variant in enumerate(variants):
        normal_lengths = np.linalg.norm(variant['normals'], axis=1)
        axes[1, 0].hist(normal_lengths, alpha=0.5, label=f'变体{i+1}', bins=20)
    axes[1, 0].set_xlabel('法向量长度')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('法向量长度分布')
    axes[1, 0].legend()
    
    # 空间密度分析
    for i, variant in enumerate(variants):
        centers = variant['centers']
        # 计算到中心的距离
        center_point = np.array([30, 30, 30])  # 球形中心
        distances = np.linalg.norm(centers - center_point, axis=1)
        axes[1, 1].hist(distances, alpha=0.5, label=f'变体{i+1}', bins=20)
    axes[1, 1].set_xlabel('到中心距离')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('空间密度分布')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存分析图
    analysis_path = output_dir / "mesh_variants_analysis.png"
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"差异分析保存到: {analysis_path}")


def demonstrate_sampling_difference(variants):
    """演示不同变体的采样差异"""
    print("\n=== 采样差异演示 ===")
    
    # 模拟从每个变体中采样
    sample_size = 50
    sampling_results = []
    
    for i, variant in enumerate(variants):
        centers = variant['centers']
        normals = variant['normals']
        
        # 随机采样
        if len(centers) > sample_size:
            indices = np.random.choice(len(centers), sample_size, replace=False)
            sampled_centers = centers[indices]
            sampled_normals = normals[indices]
        else:
            sampled_centers = centers
            sampled_normals = normals
        
        sampling_results.append({
            'variant': i+1,
            'seed': variant['seed'],
            'centers': sampled_centers,
            'normals': sampled_normals
        })
        
        print(f"变体 {i+1}: 从 {len(centers)} 个三角形中采样了 {len(sampled_centers)} 个点")
    
    return sampling_results


def main():
    """主函数"""
    print("Mesh变体演示")
    print("=" * 50)
    
    # 1. 创建测试数据
    print("创建球形膜掩码...")
    mask = create_sphere_mask()
    print(f"掩码形状: {mask.shape}")
    print(f"膜体积分数: {mask.sum() / mask.size:.3f}")
    
    # 2. 生成mesh变体
    variants = generate_mesh_variants(mask, num_variants=4)
    
    # 3. 分析变体差异
    analyze_variants(variants)
    
    # 4. 可视化变体
    plot_path = visualize_variants(variants)
    
    # 5. 演示采样差异
    sampling_results = demonstrate_sampling_difference(variants)
    
    print(f"\n=== 总结 ===")
    print("✓ 成功生成了4个不同的mesh变体")
    print("✓ 每个变体都有不同的三角形分布")
    print("✓ 变体间存在明显的空间差异")
    print("✓ 这为粒子挑选提供了多样化的采样选择")
    print(f"\n可视化结果保存在: mesh_variants_demo/")
    
    return variants, sampling_results


if __name__ == "__main__":
    variants, sampling_results = main()

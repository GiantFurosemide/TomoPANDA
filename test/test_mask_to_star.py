#!/usr/bin/env python3
"""
测试从 mask MRC 文件生成 particle.star 文件
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tomopanda.utils.mrc_utils import MRCWriter
from tomopanda.core.voxel_sample import run_voxel_sampling_to_star

def create_test_mask():
    """创建测试 mask"""
    print("=== 创建测试 mask ===")
    
    # 创建 (100, 100, 50) 的 mask
    mask = np.zeros((100, 100, 50), dtype=np.uint8)
    
    # 在 XY=(50,50) 的平面设置为1
    mask[50, 50, :] = 1
    
    print(f"Mask 形状: {mask.shape}")
    print(f"Mask 中值为1的体素数量: {np.sum(mask)}")
    print(f"Mask 中值为1的位置: {np.where(mask == 1)}")
    
    return mask

def save_mask_as_mrc(mask, output_path):
    """保存 mask 为 MRC 文件"""
    print(f"\n=== 保存 mask 为 MRC 文件 ===")
    
    # 转换为 float32 用于 MRC 写入
    mask_float = mask.astype(np.float32)
    
    # 写入 MRC 文件
    MRCWriter.write_mrc(mask_float, output_path)
    print(f"保存 MRC 文件: {output_path}")
    print(f"文件大小: {Path(output_path).stat().st_size} bytes")

def generate_particles_from_mask(mask, output_star):
    """从 mask 生成粒子 STAR 文件"""
    print(f"\n=== 从 mask 生成粒子 STAR 文件 ===")
    
    # 使用 voxel sampling 生成粒子
    centers, normals = run_voxel_sampling_to_star(
        mask=mask,
        min_distance=5.0,  # 最小距离5像素
        edge_distance=2.0,  # 边缘距离2像素
        output_path=output_star,
        tomogram_name="test_tomogram",
        particle_diameter=200.0,
        voxel_size=(1.0, 1.0, 1.0)  # 体素大小
    )
    
    print(f"生成粒子数量: {len(centers)}")
    print(f"保存 STAR 文件: {output_star}")
    
    if len(centers) > 0:
        print("\n前5个粒子的坐标和法向量:")
        for i, (center, normal) in enumerate(zip(centers[:5], normals[:5])):
            print(f"  粒子 {i+1}: center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), normal=({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    
    return centers, normals

def verify_star_file(star_file):
    """验证 STAR 文件内容"""
    print(f"\n=== 验证 STAR 文件内容 ===")
    
    if not Path(star_file).exists():
        print(f"STAR 文件不存在: {star_file}")
        return
    
    with open(star_file, 'r') as f:
        lines = f.readlines()
    
    print(f"STAR 文件总行数: {len(lines)}")
    
    # 找到数据行
    data_lines = []
    in_data_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('_rlnCoordinateX'):
            in_data_section = True
            continue
        if in_data_section and line and not line.startswith('_'):
            data_lines.append(line)
    
    print(f"数据行数: {len(data_lines)}")
    
    if data_lines:
        print("\n前3行数据:")
        for i, line in enumerate(data_lines[:3]):
            parts = line.split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                print(f"  行 {i+1}: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

def main():
    """主函数"""
    print("=== 测试 mask 到 particle.star 转换 ===\n")
    
    # 创建测试目录
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)
    print(f"创建测试目录: {test_dir}")
    
    # 创建测试 mask
    mask = create_test_mask()
    
    # 保存为 MRC 文件
    mrc_file = test_dir / "test_mask.mrc"
    save_mask_as_mrc(mask, mrc_file)
    
    # 生成粒子 STAR 文件
    star_file = test_dir / "test_particles.star"
    centers, normals = generate_particles_from_mask(mask, star_file)
    
    # 验证 STAR 文件
    verify_star_file(star_file)
    
    print(f"\n=== 测试完成 ===")
    print(f"生成的文件:")
    print(f"  MRC 文件: {mrc_file}")
    print(f"  STAR 文件: {star_file}")
    
    # 询问是否清理文件
    response = input("\n是否清理测试文件? (y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        for file_path in [mrc_file, star_file]:
            if file_path.exists():
                file_path.unlink()
                print(f"删除文件: {file_path}")
        # 删除测试目录（如果为空）
        try:
            test_dir.rmdir()
            print(f"删除测试目录: {test_dir}")
        except OSError:
            print(f"测试目录不为空，保留: {test_dir}")

if __name__ == "__main__":
    main()

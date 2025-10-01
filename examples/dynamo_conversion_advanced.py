#!/usr/bin/env python3
"""
Dynamo-RELION 转换高级示例

这个示例展示了高级功能，包括自定义矩阵映射、欧拉角转换、
以及批量处理多个文件。
"""

import sys
import os
from pathlib import Path
import numpy as np
import glob

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tomopanda.utils.dynamo_relion_convert_util import (
    convert_matrix_to_tbl,
    DynamoConverter
)
from tomopanda.utils.relion_utils import RELIONConverter


def create_multiple_star_files():
    """
    创建多个示例 STAR 文件用于批量处理
    """
    print("创建多个示例 STAR 文件...")
    
    # 创建不同的 tomogram 数据
    tomogram_data = [
        {
            'name': 'tomogram_001',
            'centers': np.array([
                [100.0, 200.0, 300.0],
                [150.0, 250.0, 350.0],
                [200.0, 300.0, 400.0]
            ]),
            'normals': np.array([
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.99],
                [0.2, 0.2, 0.96]
            ])
        },
        {
            'name': 'tomogram_002',
            'centers': np.array([
                [300.0, 400.0, 500.0],
                [350.0, 450.0, 550.0]
            ]),
            'normals': np.array([
                [0.3, 0.3, 0.91],
                [0.4, 0.4, 0.84]
            ])
        }
    ]
    
    star_files = []
    for i, data in enumerate(tomogram_data):
        star_file = f"sample_tomogram_{i+1:03d}.star"
        RELIONConverter.create_star_file(
            centers=data['centers'],
            normals=data['normals'],
            output_path=star_file,
            tomogram_name=data['name'],
            particle_diameter=200.0,
            confidence=0.8
        )
        star_files.append(star_file)
        print(f"  创建: {star_file}")
    
    return star_files


def custom_matrix_mapping_example():
    """
    自定义矩阵映射示例
    """
    print("\n=== 自定义矩阵映射示例 ===\n")
    
    # 创建自定义数据矩阵
    # 每行包含: x, y, z, phi, theta, psi
    data = np.array([
        [100.0, 200.0, 300.0, 10.0, 20.0, 30.0],
        [150.0, 250.0, 350.0, 15.0, 25.0, 35.0],
        [200.0, 300.0, 400.0, 20.0, 30.0, 40.0],
        [250.0, 350.0, 450.0, 25.0, 35.0, 45.0],
        [300.0, 400.0, 500.0, 30.0, 40.0, 50.0]
    ])
    
    print(f"输入数据矩阵形状: {data.shape}")
    print("数据内容:")
    for i, row in enumerate(data):
        print(f"  粒子 {i+1}: x={row[0]:.1f}, y={row[1]:.1f}, z={row[2]:.1f}, "
              f"phi={row[3]:.1f}, theta={row[4]:.1f}, psi={row[5]:.1f}")
    
    # 映射到 Dynamo 列: 坐标到列24-26，角度到列7-9
    target_columns = [24, 25, 26, 7, 8, 9]  # x, y, z, phi, theta, psi
    
    print(f"\n映射到 Dynamo 列: {target_columns}")
    
    convert_matrix_to_tbl(
        matrix=data,
        target_columns=target_columns,
        output_path="custom_matrix.tbl",
        start_tag=1
    )
    
    # 显示结果
    if Path("custom_matrix.tbl").exists():
        print("\n生成的 .tbl 文件内容 (前3行):")
        with open("custom_matrix.tbl", 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"   {i+1}: {line.strip()}")


def euler_angle_conversion_example():
    """
    欧拉角转换示例
    """
    print("\n=== 欧拉角转换示例 ===\n")
    
    # 测试不同的 RELION 角度
    test_angles = [
        (0.0, 0.0, 0.0),      # 无旋转
        (30.0, 45.0, 60.0),    # 小角度
        (90.0, 90.0, 90.0),    # 大角度
        (180.0, 0.0, 0.0),     # 180度旋转
        (45.0, 135.0, 225.0)  # 复杂角度
    ]
    
    print("RELION 到 Dynamo 欧拉角转换:")
    print("RELION (tilt, psi, rot) -> Dynamo (phi, theta, psi)")
    print("-" * 60)
    
    for tilt, psi, rot in test_angles:
        phi, theta, psi_dynamo = DynamoConverter.relion_to_dynamo_euler(tilt, psi, rot)
        print(f"({tilt:6.1f}, {psi:6.1f}, {rot:6.1f}) -> "
              f"({phi:6.1f}, {theta:6.1f}, {psi_dynamo:6.1f})")
    
    print("\n角度范围检查:")
    print("Dynamo 角度范围:")
    print("  phi: [-180, +180]")
    print("  theta: [0, 180]")
    print("  psi: [-180, +180]")


def batch_processing_example():
    """
    批量处理示例
    """
    print("\n=== 批量处理示例 ===\n")
    
    # 创建多个 STAR 文件
    star_files = create_multiple_star_files()
    
    # 批量转换
    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n批量转换到目录: {output_dir}")
    
    for i, star_file in enumerate(star_files):
        base_name = Path(star_file).stem
        print(f"处理文件 {i+1}/{len(star_files)}: {star_file}")
        
        try:
            from tomopanda.utils.dynamo_relion_convert_util import convert_relion_to_dynamo
            
            tbl_path, vll_path = convert_relion_to_dynamo(
                star_file=star_file,
                output_dir=output_dir,
                base_name=base_name,
                start_tag=i * 1000 + 1,  # 避免标签冲突
                model_index=1
            )
            print(f"  输出: {tbl_path.name}, {vll_path.name}")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    # 显示批量处理结果
    print(f"\n批量处理结果:")
    tbl_files = list(output_dir.glob("*.tbl"))
    vll_files = list(output_dir.glob("*.vll"))
    
    print(f"  生成的 .tbl 文件: {len(tbl_files)}")
    print(f"  生成的 .vll 文件: {len(vll_files)}")
    
    for tbl_file in tbl_files:
        print(f"    {tbl_file.name}")


def advanced_matrix_operations():
    """
    高级矩阵操作示例
    """
    print("\n=== 高级矩阵操作示例 ===\n")
    
    # 创建复杂的粒子数据
    n_particles = 10
    np.random.seed(42)  # 确保可重复性
    
    # 生成随机坐标
    coordinates = np.random.uniform(50, 500, (n_particles, 3))
    
    # 生成随机欧拉角
    phi = np.random.uniform(-180, 180, n_particles)
    theta = np.random.uniform(0, 180, n_particles)
    psi = np.random.uniform(-180, 180, n_particles)
    
    # 组合数据矩阵
    data = np.column_stack([coordinates, phi, theta, psi])
    
    print(f"生成 {n_particles} 个随机粒子:")
    print("坐标范围: [50, 500]")
    print("角度范围: phi[-180,180], theta[0,180], psi[-180,180]")
    
    # 不同的映射策略
    mapping_strategies = [
        {
            'name': '标准映射',
            'columns': [24, 25, 26, 7, 8, 9],  # x, y, z, phi, theta, psi
            'output': 'standard_mapping.tbl'
        },
        {
            'name': '仅坐标映射',
            'columns': [24, 25, 26],  # 只映射坐标
            'output': 'coordinates_only.tbl'
        },
        {
            'name': '仅角度映射',
            'columns': [7, 8, 9],  # 只映射角度
            'output': 'angles_only.tbl'
        }
    ]
    
    for strategy in mapping_strategies:
        print(f"\n{strategy['name']}:")
        print(f"  映射列: {strategy['columns']}")
        
        # 选择对应的数据列
        if len(strategy['columns']) == 6:
            matrix_data = data
        elif len(strategy['columns']) == 3:
            if strategy['columns'] == [24, 25, 26]:
                matrix_data = data[:, :3]  # 只取坐标
            else:
                matrix_data = data[:, 3:]  # 只取角度
        else:
            continue
        
        convert_matrix_to_tbl(
            matrix=matrix_data,
            target_columns=strategy['columns'],
            output_path=strategy['output'],
            start_tag=1
        )
        
        print(f"  输出文件: {strategy['output']}")


def cleanup_files():
    """
    清理临时文件
    """
    files_to_remove = [
        "custom_matrix.tbl",
        "standard_mapping.tbl",
        "coordinates_only.tbl",
        "angles_only.tbl"
    ]
    
    # 删除示例 STAR 文件
    star_files = glob.glob("sample_tomogram_*.star")
    files_to_remove.extend(star_files)
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"删除临时文件: {file_path}")
    
    # 删除输出目录
    if Path("batch_output").exists():
        import shutil
        shutil.rmtree("batch_output")
        print("删除输出目录: batch_output")


def main():
    """
    主函数
    """
    try:
        # 自定义矩阵映射示例
        custom_matrix_mapping_example()
        
        # 欧拉角转换示例
        euler_angle_conversion_example()
        
        # 批量处理示例
        batch_processing_example()
        
        # 高级矩阵操作示例
        advanced_matrix_operations()
        
        print("\n=== 高级示例完成 ===")
        print("\n这些示例展示了 Dynamo-RELION 转换工具的高级功能。")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 询问是否清理文件
        response = input("\n是否清理临时文件? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            cleanup_files()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

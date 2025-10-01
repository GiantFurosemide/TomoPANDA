#!/usr/bin/env python3
"""
Dynamo-RELION 转换基本示例

这个示例展示了如何使用 TomoPANDA 的 Dynamo-RELION 转换工具
将 RELION STAR 文件转换为 Dynamo 格式。
"""

import sys
import os
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tomopanda.utils.dynamo_relion_convert_util import (
    relion_star_to_dynamo_tbl_vll,
    convert_relion_to_dynamo,
    DynamoConverter
)
from tomopanda.utils.relion_utils import RELIONConverter


def create_sample_star_file(output_path: str) -> None:
    """
    创建一个示例 RELION STAR 文件用于测试
    """
    # 创建示例粒子数据
    centers = np.array([
        [100.0, 200.0, 300.0],
        [150.0, 250.0, 350.0],
        [200.0, 300.0, 400.0],
        [250.0, 350.0, 450.0],
        [300.0, 400.0, 500.0]
    ])
    
    # 创建示例法向量
    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 0.99],
        [0.2, 0.2, 0.96],
        [0.3, 0.3, 0.91],
        [0.4, 0.4, 0.84]
    ])
    
    # 创建 STAR 文件
    RELIONConverter.create_star_file(
        centers=centers,
        normals=normals,
        output_path=output_path,
        tomogram_name="sample_tomogram",
        particle_diameter=200.0,
        confidence=0.8
    )
    
    print(f"创建示例 STAR 文件: {output_path}")


def basic_conversion_example():
    """
    基本转换示例
    """
    print("=== Dynamo-RELION 基本转换示例 ===\n")
    
    # 创建示例 STAR 文件
    star_file = "sample_particles.star"
    create_sample_star_file(star_file)
    
    # 方法1: 直接转换
    print("1. 直接转换方法:")
    relion_star_to_dynamo_tbl_vll(
        star_file=star_file,
        tbl_output="particles.tbl",
        vll_output="particles.vll",
        start_tag=1,
        model_index=1
    )
    print("   输出文件: particles.tbl, particles.vll\n")
    
    # 方法2: 便捷转换函数
    print("2. 便捷转换方法:")
    tbl_path, vll_path = convert_relion_to_dynamo(
        star_file=star_file,
        output_dir="dynamo_output",
        base_name="converted_particles",
        start_tag=1,
        model_index=1
    )
    print(f"   输出文件: {tbl_path}, {vll_path}\n")
    
    # 显示文件内容
    print("3. 查看输出文件内容:")
    
    # 显示 .tbl 文件内容
    if Path("particles.tbl").exists():
        print("   .tbl 文件内容 (前3行):")
        with open("particles.tbl", 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"   {i+1}: {line.strip()}")
        print()
    
    # 显示 .vll 文件内容
    if Path("particles.vll").exists():
        print("   .vll 文件内容:")
        with open("particles.vll", 'r') as f:
            for line in f:
                print(f"   {line.strip()}")
        print()


def custom_parameters_example():
    """
    自定义参数示例
    """
    print("=== 自定义参数示例 ===\n")
    
    star_file = "sample_particles.star"
    
    # 使用不同的起始标签和模型索引
    print("使用自定义参数:")
    relion_star_to_dynamo_tbl_vll(
        star_file=star_file,
        tbl_output="custom_particles.tbl",
        vll_output="custom_particles.vll",
        start_tag=100,  # 从标签100开始
        model_index=2   # 模型索引为2
    )
    
    # 显示结果
    if Path("custom_particles.tbl").exists():
        print("   自定义 .tbl 文件内容 (前2行):")
        with open("custom_particles.tbl", 'r') as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                print(f"   {i+1}: {line.strip()}")
        print()


def cleanup_files():
    """
    清理临时文件
    """
    files_to_remove = [
        "sample_particles.star",
        "particles.tbl",
        "particles.vll",
        "custom_particles.tbl",
        "custom_particles.vll"
    ]
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"删除临时文件: {file_path}")
    
    # 删除输出目录
    if Path("dynamo_output").exists():
        import shutil
        shutil.rmtree("dynamo_output")
        print("删除输出目录: dynamo_output")


def main():
    """
    主函数
    """
    try:
        # 基本转换示例
        basic_conversion_example()
        
        # 自定义参数示例
        custom_parameters_example()
        
        print("=== 示例完成 ===")
        print("\n注意: 这些是示例文件，实际使用时请使用真实的 RELION STAR 文件。")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    finally:
        # 询问是否清理文件
        response = input("\n是否清理临时文件? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            cleanup_files()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

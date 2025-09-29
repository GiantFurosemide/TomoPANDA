#!/usr/bin/env python3
"""
TomoPANDA基本使用示例
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """运行命令并显示输出"""
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"返回码: {result.returncode}")
    if result.stdout:
        print(f"输出:\n{result.stdout}")
    if result.stderr:
        print(f"错误:\n{result.stderr}")
    return result.returncode

def main():
    """基本使用示例"""
    print("=== TomoPANDA 基本使用示例 ===\n")
    
    # 1. 显示版本信息
    print("1. 显示版本信息:")
    run_command(["python", "-m", "tomopanda.cli.main", "version"])
    print()
    
    # 2. 初始化配置
    print("2. 初始化配置文件:")
    run_command(["python", "-m", "tomopanda.cli.main", "config", "init", "--template", "detect"])
    print()
    
    # 3. 显示配置
    print("3. 显示当前配置:")
    run_command(["python", "-m", "tomopanda.cli.main", "config", "show"])
    print()
    
    # 4. 显示帮助信息
    print("4. 显示帮助信息:")
    run_command(["python", "-m", "tomopanda.cli.main", "--help"])
    print()
    
    # 5. 显示检测命令帮助
    print("5. 显示检测命令帮助:")
    run_command(["python", "-m", "tomopanda.cli.main", "detect", "--help"])
    print()

    # 6. 运行sample voxel-sample（合成数据）
    print("6. 运行sample voxel-sample（合成数据）:")
    run_command([
        "python", "-m", "tomopanda.cli.main", "sample", "voxel-sample",
        "--create-synthetic", "--output", "examples_voxel_results"
    ])
    print()

    # 7. 运行sample mesh-geodesic（合成数据）
    print("7. 运行sample mesh-geodesic（合成数据）:")
    run_command([
        "python", "-m", "tomopanda.cli.main", "sample", "mesh-geodesic",
        "--create-synthetic", "--output", "examples_mesh_results"
    ])
    print()
    
    print("=== 示例完成 ===")

if __name__ == "__main__":
    main()

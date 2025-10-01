#!/usr/bin/env python3
"""
Dynamo-RELION 批量转换示例

这个示例展示了如何批量处理多个 RELION STAR 文件，
包括文件管理、错误处理和进度跟踪。
"""

import sys
import os
from pathlib import Path
import numpy as np
import glob
import time
from typing import List, Tuple, Dict
import json

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tomopanda.utils.dynamo_relion_convert_util import (
    convert_relion_to_dynamo,
    relion_star_to_dynamo_tbl_vll
)
from tomopanda.utils.relion_utils import RELIONConverter


def create_batch_test_data(num_tomograms: int = 5, particles_per_tomo: int = 10) -> List[str]:
    """
    创建批量测试数据
    
    Args:
        num_tomograms: tomogram 数量
        particles_per_tomo: 每个 tomogram 的粒子数量
        
    Returns:
        创建的 STAR 文件路径列表
    """
    print(f"创建批量测试数据: {num_tomograms} 个 tomogram，每个 {particles_per_tomo} 个粒子")
    
    star_files = []
    np.random.seed(42)  # 确保可重复性
    
    for i in range(num_tomograms):
        # 生成随机坐标
        centers = np.random.uniform(50, 500, (particles_per_tomo, 3))
        
        # 生成随机法向量
        normals = np.random.randn(particles_per_tomo, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        # 创建 STAR 文件
        star_file = f"batch_tomogram_{i+1:03d}.star"
        RELIONConverter.create_star_file(
            centers=centers,
            normals=normals,
            output_path=star_file,
            tomogram_name=f"tomogram_{i+1:03d}",
            particle_diameter=200.0,
            confidence=0.8
        )
        star_files.append(star_file)
        print(f"  创建: {star_file}")
    
    return star_files


def batch_convert_simple(star_files: List[str], output_dir: str = "batch_output") -> Dict[str, any]:
    """
    简单批量转换
    
    Args:
        star_files: STAR 文件路径列表
        output_dir: 输出目录
        
    Returns:
        转换结果统计
    """
    print(f"\n=== 简单批量转换 ===")
    print(f"处理 {len(star_files)} 个文件")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'success': 0,
        'failed': 0,
        'files': [],
        'start_time': time.time()
    }
    
    for i, star_file in enumerate(star_files):
        print(f"处理 {i+1}/{len(star_files)}: {star_file}")
        
        try:
            base_name = Path(star_file).stem
            tbl_path, vll_path = convert_relion_to_dynamo(
                star_file=star_file,
                output_dir=output_dir,
                base_name=base_name,
                start_tag=i * 1000 + 1,  # 避免标签冲突
                model_index=1
            )
            
            results['success'] += 1
            results['files'].append({
                'input': star_file,
                'tbl': str(tbl_path),
                'vll': str(vll_path),
                'status': 'success'
            })
            print(f"  成功: {tbl_path.name}, {vll_path.name}")
            
        except Exception as e:
            results['failed'] += 1
            results['files'].append({
                'input': star_file,
                'status': 'failed',
                'error': str(e)
            })
            print(f"  失败: {e}")
    
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']
    
    return results


def batch_convert_with_progress(star_files: List[str], output_dir: str = "batch_output_progress") -> Dict[str, any]:
    """
    带进度跟踪的批量转换
    
    Args:
        star_files: STAR 文件路径列表
        output_dir: 输出目录
        
    Returns:
        转换结果统计
    """
    print(f"\n=== 带进度跟踪的批量转换 ===")
    print(f"处理 {len(star_files)} 个文件")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'success': 0,
        'failed': 0,
        'files': [],
        'start_time': time.time(),
        'progress': []
    }
    
    for i, star_file in enumerate(star_files):
        start_time = time.time()
        progress = (i / len(star_files)) * 100
        
        print(f"[{progress:5.1f}%] 处理 {i+1}/{len(star_files)}: {star_file}")
        
        try:
            base_name = Path(star_file).stem
            tbl_path, vll_path = convert_relion_to_dynamo(
                star_file=star_file,
                output_dir=output_dir,
                base_name=base_name,
                start_tag=i * 1000 + 1,
                model_index=1
            )
            
            processing_time = time.time() - start_time
            results['success'] += 1
            results['files'].append({
                'input': star_file,
                'tbl': str(tbl_path),
                'vll': str(vll_path),
                'status': 'success',
                'processing_time': processing_time
            })
            results['progress'].append({
                'file': star_file,
                'progress': progress,
                'time': processing_time,
                'status': 'success'
            })
            print(f"  成功 ({processing_time:.2f}s): {tbl_path.name}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            results['failed'] += 1
            results['files'].append({
                'input': star_file,
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time
            })
            results['progress'].append({
                'file': star_file,
                'progress': progress,
                'time': processing_time,
                'status': 'failed'
            })
            print(f"  失败 ({processing_time:.2f}s): {e}")
    
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']
    
    return results


def batch_convert_with_validation(star_files: List[str], output_dir: str = "batch_output_validation") -> Dict[str, any]:
    """
    带验证的批量转换
    
    Args:
        star_files: STAR 文件路径列表
        output_dir: 输出目录
        
    Returns:
        转换结果统计
    """
    print(f"\n=== 带验证的批量转换 ===")
    print(f"处理 {len(star_files)} 个文件")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'success': 0,
        'failed': 0,
        'files': [],
        'start_time': time.time(),
        'validation': []
    }
    
    for i, star_file in enumerate(star_files):
        print(f"处理 {i+1}/{len(star_files)}: {star_file}")
        
        try:
            # 验证输入文件
            if not Path(star_file).exists():
                raise FileNotFoundError(f"文件不存在: {star_file}")
            
            file_size = Path(star_file).stat().st_size
            if file_size == 0:
                raise ValueError(f"文件为空: {star_file}")
            
            base_name = Path(star_file).stem
            tbl_path, vll_path = convert_relion_to_dynamo(
                star_file=star_file,
                output_dir=output_dir,
                base_name=base_name,
                start_tag=i * 1000 + 1,
                model_index=1
            )
            
            # 验证输出文件
            validation_result = validate_output_files(tbl_path, vll_path)
            
            results['success'] += 1
            results['files'].append({
                'input': star_file,
                'tbl': str(tbl_path),
                'vll': str(vll_path),
                'status': 'success',
                'validation': validation_result
            })
            results['validation'].append(validation_result)
            print(f"  成功: {tbl_path.name}, {vll_path.name}")
            print(f"    验证: {validation_result['summary']}")
            
        except Exception as e:
            results['failed'] += 1
            results['files'].append({
                'input': star_file,
                'status': 'failed',
                'error': str(e)
            })
            print(f"  失败: {e}")
    
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']
    
    return results


def validate_output_files(tbl_path: Path, vll_path: Path) -> Dict[str, any]:
    """
    验证输出文件
    
    Args:
        tbl_path: .tbl 文件路径
        vll_path: .vll 文件路径
        
    Returns:
        验证结果
    """
    validation = {
        'tbl_valid': False,
        'vll_valid': False,
        'tbl_rows': 0,
        'vll_rows': 0,
        'summary': ''
    }
    
    # 验证 .tbl 文件
    if tbl_path.exists():
        with open(tbl_path, 'r') as f:
            lines = f.readlines()
            validation['tbl_rows'] = len(lines)
            if lines:
                # 检查列数
                first_line_cols = len(lines[0].strip().split())
                validation['tbl_valid'] = first_line_cols == 35
                validation['summary'] += f"TBL: {validation['tbl_rows']} 行, {first_line_cols} 列"
    
    # 验证 .vll 文件
    if vll_path.exists():
        with open(vll_path, 'r') as f:
            lines = f.readlines()
            validation['vll_rows'] = len(lines)
            validation['vll_valid'] = len(lines) > 0
            validation['summary'] += f", VLL: {validation['vll_rows']} 行"
    
    return validation


def save_batch_results(results: Dict[str, any], output_file: str = "batch_results.json") -> None:
    """
    保存批量处理结果
    
    Args:
        results: 处理结果
        output_file: 输出文件路径
    """
    # 准备可序列化的结果
    serializable_results = {
        'summary': {
            'total_files': len(results['files']),
            'success': results['success'],
            'failed': results['failed'],
            'duration': results['duration']
        },
        'files': results['files']
    }
    
    # 添加进度信息（如果存在）
    if 'progress' in results:
        serializable_results['progress'] = results['progress']
    
    # 添加验证信息（如果存在）
    if 'validation' in results:
        serializable_results['validation'] = results['validation']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"保存结果到: {output_file}")


def print_batch_summary(results: Dict[str, any]) -> None:
    """
    打印批量处理摘要
    
    Args:
        results: 处理结果
    """
    print(f"\n=== 批量处理摘要 ===")
    print(f"总文件数: {len(results['files'])}")
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    print(f"总耗时: {results['duration']:.2f} 秒")
    
    if results['success'] > 0:
        avg_time = results['duration'] / results['success']
        print(f"平均处理时间: {avg_time:.2f} 秒/文件")
    
    # 显示失败的文件
    failed_files = [f for f in results['files'] if f['status'] == 'failed']
    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  {f['input']}: {f.get('error', '未知错误')}")


def cleanup_files():
    """
    清理临时文件
    """
    files_to_remove = [
        "batch_results.json"
    ]
    
    # 删除示例 STAR 文件
    star_files = glob.glob("batch_tomogram_*.star")
    files_to_remove.extend(star_files)
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"删除临时文件: {file_path}")
    
    # 删除输出目录
    output_dirs = ["batch_output", "batch_output_progress", "batch_output_validation"]
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"删除输出目录: {output_dir}")


def main():
    """
    主函数
    """
    try:
        # 创建测试数据
        star_files = create_batch_test_data(num_tomograms=5, particles_per_tomo=8)
        
        # 简单批量转换
        results1 = batch_convert_simple(star_files[:3], "batch_output")
        print_batch_summary(results1)
        save_batch_results(results1, "batch_results_simple.json")
        
        # 带进度跟踪的批量转换
        results2 = batch_convert_with_progress(star_files[3:], "batch_output_progress")
        print_batch_summary(results2)
        save_batch_results(results2, "batch_results_progress.json")
        
        # 带验证的批量转换
        results3 = batch_convert_with_validation(star_files, "batch_output_validation")
        print_batch_summary(results3)
        save_batch_results(results3, "batch_results_validation.json")
        
        print("\n=== 批量转换示例完成 ===")
        print("这些示例展示了不同的批量处理策略和错误处理方法。")
        
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

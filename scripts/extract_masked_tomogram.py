#!/usr/bin/env python3
"""
使用 mrcfile 和 numpy 进行mask扩展和tomogram提取
支持高斯模糊mask边界和添加高斯白噪声背景
"""

import os
import sys
import argparse
import numpy as np
import mrcfile
from scipy import ndimage
from typing import Tuple, Optional
import warnings

def read_mrc_file(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    读取MRC文件并返回数据和元数据
    
    Args:
        filepath (str): MRC文件路径
        
    Returns:
        Tuple[np.ndarray, dict]: (数据数组, 元数据字典)
            - 数据数组: 3D numpy数组 (z, y, x)
            - 元数据字典: 包含像素大小、数据类型等信息
    """
    with mrcfile.open(filepath, mode='r') as mrc:
        data = mrc.data.copy()
        metadata = {
            'voxel_size': mrc.voxel_size,
            'nx': mrc.header.nx,
            'ny': mrc.header.ny,
            'nz': mrc.header.nz,
            'mode': mrc.header.mode,
            'dtype': data.dtype
        }
    return data, metadata

def write_mrc_file(filepath: str, data: np.ndarray, metadata: dict, 
                   overwrite: bool = False) -> None:
    """
    写入MRC文件
    
    Args:
        filepath (str): 输出文件路径
        data (np.ndarray): 3D数据数组 (z, y, x)
        metadata (dict): 元数据字典
        overwrite (bool): 是否覆盖已存在的文件
    """
    with mrcfile.new(filepath, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = metadata['voxel_size']
        mrc.update_header_from_data()

def expand_mask_simple(mask: np.ndarray, expand_pixels: int = 5, 
                      gaussian_sigma: float = 1.0) -> tuple:
    """
    简单直接的mask扩展：先形态学扩展，再高斯平滑边界
    
    Args:
        mask (np.ndarray): 输入mask数组
        expand_pixels (int): 扩展像素数（所见即所得）
        gaussian_sigma (float): 边界高斯平滑标准差
        
    Returns:
        tuple: (扩展后的mask, 分析结果字典)
    """
    import time
    from scipy import ndimage
    
    print(f"开始mask扩展处理...")
    start_time = time.time()
    
    # 步骤1: 标准化mask到0-1范围
    print("步骤1: 标准化mask...")
    mask_normalized = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask_normalized = mask_normalized / 255.0
    
    # 计算原始体积
    original_volume = np.sum(mask_normalized > 0.5)
    print(f"原始mask体积: {original_volume:,} 体素")
    
    # 步骤2: 形态学扩展（所见即所得）
    print(f"步骤2: 形态学扩展 {expand_pixels} 像素...")
    structure_size = expand_pixels*2+1
    print(f"结构元素大小: {structure_size}×{structure_size}×{structure_size}")
    
    # 对于大扩展像素，使用迭代扩展以节省内存
    if expand_pixels > 10:
        print("使用迭代扩展以优化内存使用...")
        expanded_mask = mask_normalized > 0.5
        structure_small = np.ones((3, 3, 3), dtype=np.uint8)
        
        for i in range(expand_pixels):
            if i % 5 == 0:  # 每5次迭代显示进度
                print(f"  扩展进度: {i+1}/{expand_pixels}")
            expanded_mask = ndimage.binary_dilation(expanded_mask, structure=structure_small)
    else:
        # 小扩展像素直接使用大结构元素
        structure = np.ones((structure_size, structure_size, structure_size), dtype=np.uint8)
        expanded_mask = ndimage.binary_dilation(mask_normalized > 0.5, structure=structure)
    
    expanded_mask = expanded_mask.astype(np.float32)
    dilation_time = time.time() - start_time
    print(f"形态学扩展完成，耗时: {dilation_time:.2f}秒")
    
    # 步骤3: 高斯平滑边界
    if gaussian_sigma > 0:
        print(f"步骤3: 高斯平滑边界 (sigma={gaussian_sigma})...")
        gaussian_start = time.time()
        # 对扩展后的mask进行高斯模糊
        blurred_mask = ndimage.gaussian_filter(expanded_mask, sigma=gaussian_sigma)
        # 二值化保持扩展效果
        expanded_mask = (blurred_mask > 0.5).astype(np.float32)
        gaussian_time = time.time() - gaussian_start
        print(f"高斯平滑完成，耗时: {gaussian_time:.2f}秒")
    else:
        print("跳过高斯平滑步骤")
    
    # 步骤4: 计算统计信息
    expanded_volume = np.sum(expanded_mask > 0.5)
    actual_expansion_ratio = expanded_volume / original_volume if original_volume > 0 else 0
    total_time = time.time() - start_time
    
    print(f"mask扩展总耗时: {total_time:.2f}秒")
    print(f"扩展后体积: {expanded_volume:,} 体素")
    print(f"实际扩展比例: {actual_expansion_ratio:.2f}x")
    
    # 分析结果
    analysis_results = {
        'original_volume': original_volume,
        'expanded_volume': expanded_volume,
        'actual_expansion_ratio': actual_expansion_ratio,
        'expand_pixels': expand_pixels,
        'gaussian_sigma': gaussian_sigma,
        'method': 'morphological_dilation + gaussian_smoothing',
        'processing_time': total_time
    }
    
    return expanded_mask, analysis_results

def expand_mask_fast(mask: np.ndarray, expand_pixels: int = 5, 
                    gaussian_sigma: float = 1.0) -> tuple:
    """
    快速mask扩展：使用距离变换方法（更快）
    
    Args:
        mask (np.ndarray): 输入mask数组
        expand_pixels (int): 扩展像素数（所见即所得）
        gaussian_sigma (float): 边界高斯平滑标准差
        
    Returns:
        tuple: (扩展后的mask, 分析结果字典)
    """
    import time
    from scipy import ndimage
    
    print(f"开始快速mask扩展处理...")
    start_time = time.time()
    
    # 步骤1: 标准化mask到0-1范围
    print("步骤1: 标准化mask...")
    mask_normalized = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask_normalized = mask_normalized / 255.0
    
    # 计算原始体积
    original_volume = np.sum(mask_normalized > 0.5)
    print(f"原始mask体积: {original_volume:,} 体素")
    
    # 步骤2: 使用距离变换进行快速扩展
    print(f"步骤2: 距离变换扩展 {expand_pixels} 像素...")
    dist_start = time.time()
    
    # 计算到mask边界的距离
    distance = ndimage.distance_transform_edt(~(mask_normalized > 0.5))
    
    # 扩展：距离小于等于expand_pixels的区域
    expanded_mask = (distance <= expand_pixels).astype(np.float32)
    
    dist_time = time.time() - dist_start
    print(f"距离变换扩展完成，耗时: {dist_time:.2f}秒")
    
    # 步骤3: 高斯平滑边界
    if gaussian_sigma > 0:
        print(f"步骤3: 高斯平滑边界 (sigma={gaussian_sigma})...")
        gaussian_start = time.time()
        blurred_mask = ndimage.gaussian_filter(expanded_mask, sigma=gaussian_sigma)
        expanded_mask = (blurred_mask > 0.5).astype(np.float32)
        gaussian_time = time.time() - gaussian_start
        print(f"高斯平滑完成，耗时: {gaussian_time:.2f}秒")
    else:
        print("跳过高斯平滑步骤")
    
    # 步骤4: 计算统计信息
    expanded_volume = np.sum(expanded_mask > 0.5)
    actual_expansion_ratio = expanded_volume / original_volume if original_volume > 0 else 0
    total_time = time.time() - start_time
    
    print(f"快速mask扩展总耗时: {total_time:.2f}秒")
    print(f"扩展后体积: {expanded_volume:,} 体素")
    print(f"实际扩展比例: {actual_expansion_ratio:.2f}x")
    
    # 分析结果
    analysis_results = {
        'original_volume': original_volume,
        'expanded_volume': expanded_volume,
        'actual_expansion_ratio': actual_expansion_ratio,
        'expand_pixels': expand_pixels,
        'gaussian_sigma': gaussian_sigma,
        'method': 'distance_transform + gaussian_smoothing',
        'processing_time': total_time
    }
    
    return expanded_mask, analysis_results

def expand_mask_gaussian(mask: np.ndarray, sigma: float = 2.0, 
                        threshold: float = 0.5) -> np.ndarray:
    """
    使用高斯模糊扩展mask (保留向后兼容)
    
    Args:
        mask (np.ndarray): 二值mask数组 (0和1或0和255)
        sigma (float): 高斯模糊的标准差
        threshold (float): 二值化阈值
        
    Returns:
        np.ndarray: 扩展后的mask数组
    """
    # 首先将mask标准化到0-1范围
    mask_normalized = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask_normalized = mask_normalized / 255.0
    
    # 高斯模糊
    blurred_mask = ndimage.gaussian_filter(mask_normalized, sigma=sigma)
    
    # 二值化
    expanded_mask = (blurred_mask > threshold).astype(np.float32)
    
    return expanded_mask

def generate_gaussian_noise(shape: Tuple[int, ...], mean: float = 0.0, 
                           std: float = 1.0, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    生成高斯白噪声
    
    Args:
        shape (Tuple[int, ...]): 噪声数组形状
        mean (float): 噪声均值
        std (float): 噪声标准差
        dtype (np.dtype): 数据类型
        
    Returns:
        np.ndarray: 高斯噪声数组
    """
    noise = np.random.normal(mean, std, shape).astype(dtype)
    return noise

def extract_masked_tomogram(tomogram: np.ndarray, mask: np.ndarray, 
                           noise_std: float = 0.1, 
                           background_value: float = 0.0) -> np.ndarray:
    """
    提取mask区域内的tomogram并用噪声填充背景
    
    Args:
        tomogram (np.ndarray): 原始tomogram数据
        mask (np.ndarray): mask数组 (0-1之间)
        noise_std (float): 背景噪声标准差
        background_value (float): 背景基础值
        
    Returns:
        np.ndarray: 处理后的tomogram
    """
    # 确保mask和tomogram形状一致
    if tomogram.shape != mask.shape:
        raise ValueError(f"Tomogram shape {tomogram.shape} 与 mask shape {mask.shape} 不匹配")
    
    # 计算tomogram的统计信息用于噪声生成
    tomogram_mean = np.mean(tomogram)
    tomogram_std = np.std(tomogram)
    
    # 生成与tomogram统计特性匹配的噪声
    noise = generate_gaussian_noise(
        shape=tomogram.shape,
        mean=background_value,
        std=noise_std * tomogram_std,  # 相对于tomogram标准差的噪声水平
        dtype=tomogram.dtype
    )
    
    # 创建背景区域mask (1-mask)
    background_mask = 1.0 - mask
    
    # 提取mask区域内的tomogram
    masked_tomogram = tomogram * mask
    
    # 在背景区域添加噪声
    noise_background = noise * background_mask
    
    # 组合结果
    result = masked_tomogram + noise_background
    
    return result

def smooth_mask_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    平滑mask边界
    
    Args:
        mask (np.ndarray): 输入mask
        iterations (int): 平滑迭代次数
        
    Returns:
        np.ndarray: 平滑后的mask
    """
    smoothed_mask = mask.copy()
    
    for _ in range(iterations):
        # 使用形态学开运算和闭运算平滑边界
        smoothed_mask = ndimage.binary_opening(smoothed_mask, structure=np.ones((3,3,3)))
        smoothed_mask = ndimage.binary_closing(smoothed_mask, structure=np.ones((3,3,3)))
    
    return smoothed_mask.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(
        description='使用mrcfile和numpy进行mask扩展和tomogram提取',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（扩展5像素，自动选择最快方法）
  python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc
  
  # 扩展10像素，边界平滑（自动使用快速方法）
  python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --expand_pixels 10 --gaussian_sigma 1.5
  
  # 强制使用快速方法（推荐用于大扩展像素）
  python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --expand_pixels 15 --fast
  
  # 扩展20像素，无边界平滑
  python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --expand_pixels 20 --gaussian_sigma 0 --fast
        """
    )
    
    # 必需参数
    parser.add_argument('--input_tomo', required=True, help='输入tomogram文件路径 (.mrc)')
    parser.add_argument('--input_mask', required=True, help='输入mask文件路径 (.mrc)')
    parser.add_argument('--output_tomo', required=True, help='输出tomogram文件路径 (.mrc)')
    
    # 可选参数
    parser.add_argument('--sigma', type=float, default=2.0, 
                       help='高斯模糊标准差，用于mask扩展 (默认: 2.0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='mask二值化阈值 (默认: 0.5)')
    parser.add_argument('--noise_std', type=float, default=0.1,
                       help='背景噪声标准差 (相对于tomogram标准差的比例) (默认: 0.1)')
    parser.add_argument('--background_value', type=float, default=0.0,
                       help='背景基础值 (默认: 0.0)')
    parser.add_argument('--expand_pixels', type=int, default=5,
                       help='扩展像素数（所见即所得，默认: 5)')
    parser.add_argument('--gaussian_sigma', type=float, default=1.0,
                       help='边界高斯平滑标准差 (默认: 1.0，设为0禁用平滑)')
    parser.add_argument('--fast', action='store_true',
                       help='使用快速距离变换方法（推荐用于大扩展像素）')
    parser.add_argument('--smooth_iterations', type=int, default=1,
                       help='mask边界平滑迭代次数 (默认: 1)')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_tomo):
        print(f"错误: 输入tomogram文件不存在: {args.input_tomo}")
        sys.exit(1)
    if not os.path.exists(args.input_mask):
        print(f"错误: 输入mask文件不存在: {args.input_mask}")
        sys.exit(1)
    
    # 检查输出文件
    if os.path.exists(args.output_tomo) and not args.overwrite:
        print(f"错误: 输出文件已存在: {args.output_tomo}")
        print("使用 --overwrite 参数覆盖已存在的文件")
        sys.exit(1)
    
    print("开始处理...")
    print(f"输入tomogram: {args.input_tomo}")
    print(f"输入mask: {args.input_mask}")
    print(f"输出文件: {args.output_tomo}")
    
    try:
        # 步骤1: 读取输入文件
        print("步骤1: 读取输入文件...")
        tomogram, tomo_metadata = read_mrc_file(args.input_tomo)
        mask, mask_metadata = read_mrc_file(args.input_mask)
        
        print(f"Tomogram形状: {tomogram.shape}")
        print(f"Tomogram数据类型: {tomogram.dtype}")
        print(f"Tomogram像素大小: {tomo_metadata['voxel_size']}")
        print(f"Mask形状: {mask.shape}")
        print(f"Mask数据类型: {mask.dtype}")
        
        # 检查形状匹配
        if tomogram.shape != mask.shape:
            print(f"错误: Tomogram形状 {tomogram.shape} 与 mask形状 {mask.shape} 不匹配")
            sys.exit(1)
        
        # 步骤2: 扩展mask
        print("步骤2: 扩展mask...")
        
        # 显示原始mask的统计信息
        print(f"原始mask值范围: {mask.min()} - {mask.max()}")
        print(f"原始mask非零体素数: {np.sum(mask > 0)}")
        
        # 选择扩展方法
        if args.fast or args.expand_pixels > 10:
            print(f"使用快速距离变换方法")
            print(f"扩展像素数: {args.expand_pixels} (所见即所得)")
            print(f"边界高斯平滑sigma: {args.gaussian_sigma}")
            
            expanded_mask, analysis_results = expand_mask_fast(
                mask, 
                expand_pixels=args.expand_pixels,
                gaussian_sigma=args.gaussian_sigma
            )
        else:
            print(f"使用形态学扩展方法")
            print(f"扩展像素数: {args.expand_pixels} (所见即所得)")
            print(f"边界高斯平滑sigma: {args.gaussian_sigma}")
            
            expanded_mask, analysis_results = expand_mask_simple(
                mask, 
                expand_pixels=args.expand_pixels,
                gaussian_sigma=args.gaussian_sigma
            )
        
        # 显示分析结果
        print(f"实际扩展比例: {analysis_results['actual_expansion_ratio']:.2f}x")
        
        # 步骤3: 平滑mask边界
        if args.smooth_iterations > 0:
            print(f"步骤3: 平滑mask边界 (迭代次数: {args.smooth_iterations})...")
            expanded_mask = smooth_mask_boundary(expanded_mask, args.smooth_iterations)
        
        # 步骤4: 提取masked tomogram并添加噪声背景
        print("步骤4: 提取masked tomogram并添加噪声背景...")
        print(f"噪声水平: {args.noise_std} (相对于tomogram标准差)")
        print(f"背景值: {args.background_value}")
        
        result_tomogram = extract_masked_tomogram(
            tomogram=tomogram,
            mask=expanded_mask,
            noise_std=args.noise_std,
            background_value=args.background_value
        )
        
        # 步骤5: 保存结果
        print("步骤5: 保存结果...")
        write_mrc_file(args.output_tomo, result_tomogram, tomo_metadata, args.overwrite)
        
        # 显示统计信息
        print("\n处理完成！")
        print(f"输出文件: {args.output_tomo}")
        print(f"输出形状: {result_tomogram.shape}")
        print(f"输出数据类型: {result_tomogram.dtype}")
        print(f"输出像素大小: {tomo_metadata['voxel_size']}")
        
        # 计算并显示统计信息
        # 对于原始mask，检查是否需要标准化
        original_mask_normalized = mask.astype(np.float32)
        if mask.max() > 1.0:
            original_mask_normalized = original_mask_normalized / 255.0
        
        original_mask_volume = np.sum(original_mask_normalized > 0.5)
        expanded_mask_volume = np.sum(expanded_mask > 0.5)
        expansion_ratio = expanded_mask_volume / original_mask_volume if original_mask_volume > 0 else 0
        
        print(f"\n统计信息:")
        print(f"原始mask体积: {original_mask_volume} 体素")
        print(f"扩展后mask体积: {expanded_mask_volume} 体素")
        print(f"扩展比例: {expansion_ratio:.2f}x")
        print(f"结果tomogram均值: {np.mean(result_tomogram):.4f}")
        print(f"结果tomogram标准差: {np.std(result_tomogram):.4f}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 忽略numpy警告
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    main()

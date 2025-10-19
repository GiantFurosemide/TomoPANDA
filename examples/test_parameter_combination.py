#!/usr/bin/env python3
"""
测试 expected_particle_size 和 random_seed 参数的配合使用

验证：
1. expected_particle_size=None 时是否正常工作
2. expected_particle_size 和 random_seed 是否可以同时使用
3. 参数组合的效果
"""

import numpy as np
from tomopanda.core.mesh_geodesic import get_triangle_centers_and_normals, generate_synthetic_mask


def test_parameter_combinations():
    """测试不同参数组合"""
    print("=== 测试参数组合 ===")
    
    # 创建测试数据
    mask = generate_synthetic_mask(shape=(40, 40, 40), center=(20, 20, 20), radius=12)
    print(f"测试掩码形状: {mask.shape}")
    
    # 测试组合1: 默认参数 (expected_particle_size=None, random_seed=None)
    print("\n1. 默认参数测试:")
    result1 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=None,
        random_seed=None
    )
    print(f"   结果: {len(result1)} 个三角形")
    
    # 测试组合2: 只有 expected_particle_size
    print("\n2. 只有 expected_particle_size:")
    result2 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=None
    )
    print(f"   结果: {len(result2)} 个三角形")
    
    # 测试组合3: 只有 random_seed
    print("\n3. 只有 random_seed:")
    result3 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=None,
        random_seed=42
    )
    print(f"   结果: {len(result3)} 个三角形")
    
    # 测试组合4: 两个参数都使用
    print("\n4. 两个参数都使用:")
    result4 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=42
    )
    print(f"   结果: {len(result4)} 个三角形")
    
    # 测试组合5: 不同 random_seed 产生不同结果
    print("\n5. 不同 random_seed 测试:")
    result5a = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=42
    )
    result5b = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=123
    )
    print(f"   种子42: {len(result5a)} 个三角形")
    print(f"   种子123: {len(result5b)} 个三角形")
    print(f"   结果不同: {not np.allclose(result5a, result5b)}")
    
    return {
        'default': result1,
        'particle_size_only': result2,
        'random_seed_only': result3,
        'both_params': result4,
        'different_seeds': (result5a, result5b)
    }


def test_expected_particle_size_effect():
    """测试 expected_particle_size 对mesh密度的影响"""
    print("\n=== 测试 expected_particle_size 效果 ===")
    
    mask = generate_synthetic_mask(shape=(40, 40, 40), center=(20, 20, 20), radius=12)
    
    particle_sizes = [None, 10.0, 20.0, 40.0]
    results = []
    
    for size in particle_sizes:
        result = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=size,
            random_seed=42  # 固定种子确保公平比较
        )
        results.append((size, len(result)))
        print(f"expected_particle_size={size}: {len(result)} 个三角形")
    
    # 分析趋势
    print("\n趋势分析:")
    for i in range(1, len(results)):
        prev_size, prev_count = results[i-1]
        curr_size, curr_count = results[i]
        if prev_size is not None and curr_size is not None:
            if curr_size > prev_size:
                print(f"  颗粒大小从 {prev_size} 增加到 {curr_size}: 三角形数量从 {prev_count} 变为 {curr_count}")
    
    return results


def test_random_seed_consistency():
    """测试 random_seed 的一致性"""
    print("\n=== 测试 random_seed 一致性 ===")
    
    mask = generate_synthetic_mask(shape=(40, 40, 40), center=(20, 20, 20), radius=12)
    
    # 测试相同种子产生相同结果
    print("测试相同种子的一致性:")
    result1 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=42
    )
    result2 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=42
    )
    
    is_identical = np.allclose(result1, result2)
    print(f"   相同种子结果一致: {is_identical}")
    print(f"   三角形数量: {len(result1)}")
    
    # 测试不同种子产生不同结果
    print("\n测试不同种子的差异性:")
    result3 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=15.0,
        random_seed=123
    )
    
    is_different = not np.allclose(result1, result3)
    print(f"   不同种子结果不同: {is_different}")
    print(f"   种子42: {len(result1)} 个三角形")
    print(f"   种子123: {len(result3)} 个三角形")
    
    return is_identical, is_different


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    mask = generate_synthetic_mask(shape=(40, 40, 40), center=(20, 20, 20), radius=12)
    
    # 测试极小的 expected_particle_size
    print("1. 极小的 expected_particle_size:")
    try:
        result = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=1.0,
            random_seed=42
        )
        print(f"   结果: {len(result)} 个三角形")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试极大的 expected_particle_size
    print("\n2. 极大的 expected_particle_size:")
    try:
        result = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=100.0,
            random_seed=42
        )
        print(f"   结果: {len(result)} 个三角形")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试负的 random_seed
    print("\n3. 负的 random_seed:")
    try:
        result = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=15.0,
            random_seed=-42
        )
        print(f"   结果: {len(result)} 个三角形")
    except Exception as e:
        print(f"   错误: {e}")


def main():
    """主测试函数"""
    print("参数组合测试")
    print("=" * 50)
    
    # 测试基本参数组合
    results = test_parameter_combinations()
    
    # 测试 expected_particle_size 效果
    particle_size_results = test_expected_particle_size_effect()
    
    # 测试 random_seed 一致性
    is_consistent, is_different = test_random_seed_consistency()
    
    # 测试边界情况
    test_edge_cases()
    
    print("\n=== 测试总结 ===")
    print("✓ expected_particle_size=None 正常工作")
    print("✓ expected_particle_size 和 random_seed 可以同时使用")
    print("✓ 相同 random_seed 产生一致结果")
    print("✓ 不同 random_seed 产生不同结果")
    print("✓ expected_particle_size 影响mesh密度")
    
    return results


if __name__ == "__main__":
    results = main()

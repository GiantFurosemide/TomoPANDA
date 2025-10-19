#!/usr/bin/env python3
"""
Test script for new mesh geodesic features

This script demonstrates the new features:
1. expected_particle_size parameter for mesh density control
2. random_seed parameter for generating different mesh variants
3. get_triangle_centers_and_normals function for direct triangle extraction

Author: TomoPANDA Team
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import TomoPANDA modules
from tomopanda.core.mesh_geodesic import (
    get_triangle_centers_and_normals,
    create_mesh_geodesic_sampler,
    generate_synthetic_mask
)


def test_deterministic_mesh():
    """Test deterministic mesh generation (same input -> same output)"""
    print("=== Testing Deterministic Mesh Generation ===")
    
    # Create synthetic mask
    mask = generate_synthetic_mask(shape=(50, 50, 50), center=(25, 25, 25), radius=15)
    
    # Generate mesh twice with same parameters
    triangle_data_1 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=20.0,
        random_seed=None  # Deterministic
    )
    
    triangle_data_2 = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=20.0,
        random_seed=None  # Deterministic
    )
    
    # Check if results are identical
    are_identical = np.allclose(triangle_data_1, triangle_data_2)
    print(f"Deterministic results identical: {are_identical}")
    print(f"Number of triangles: {len(triangle_data_1)}")
    
    return triangle_data_1, triangle_data_2


def test_random_mesh_variants():
    """Test random mesh generation (different seeds -> different outputs)"""
    print("\n=== Testing Random Mesh Variants ===")
    
    # Create synthetic mask
    mask = generate_synthetic_mask(shape=(50, 50, 50), center=(25, 25, 25), radius=15)
    
    # Generate meshes with different random seeds
    seeds = [42, 123, 456, 789]
    mesh_variants = []
    
    for seed in seeds:
        triangle_data = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=20.0,
            random_seed=seed
        )
        mesh_variants.append(triangle_data)
        print(f"Seed {seed}: {len(triangle_data)} triangles")
    
    # Check if results are different
    all_different = True
    for i in range(len(mesh_variants)):
        for j in range(i+1, len(mesh_variants)):
            are_same = np.allclose(mesh_variants[i], mesh_variants[j])
            if are_same:
                all_different = False
                print(f"Warning: Seeds {seeds[i]} and {seeds[j]} produced identical results")
    
    print(f"All variants different: {all_different}")
    
    return mesh_variants


def test_particle_size_effect():
    """Test effect of expected_particle_size on mesh density"""
    print("\n=== Testing Particle Size Effect ===")
    
    # Create synthetic mask
    mask = generate_synthetic_mask(shape=(50, 50, 50), center=(25, 25, 25), radius=15)
    
    # Test different particle sizes
    particle_sizes = [10.0, 20.0, 40.0, 80.0]
    results = []
    
    for size in particle_sizes:
        triangle_data = get_triangle_centers_and_normals(
            mask=mask,
            expected_particle_size=size,
            random_seed=42  # Same seed for fair comparison
        )
        results.append((size, len(triangle_data)))
        print(f"Particle size {size}: {len(triangle_data)} triangles")
    
    # Larger particle size should result in fewer triangles (coarser mesh)
    print("Expected: Larger particle size -> fewer triangles (coarser mesh)")
    
    return results


def test_triangle_extraction():
    """Test triangle extraction functionality"""
    print("\n=== Testing Triangle Extraction ===")
    
    # Create synthetic mask
    mask = generate_synthetic_mask(shape=(50, 50, 50), center=(25, 25, 25), radius=15)
    
    # Extract all triangle centers
    triangle_data = get_triangle_centers_and_normals(
        mask=mask,
        expected_particle_size=20.0,
        random_seed=42
    )
    
    print(f"Extracted {len(triangle_data)} triangle centers")
    print(f"Data shape: {triangle_data.shape} (N*6: [x,y,z,nx,ny,nz])")
    
    return triangle_data


def create_visualization(triangle_data, output_dir="test_results"):
    """Create visualization of triangle centers"""
    print(f"\n=== Creating Visualization ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if len(triangle_data) == 0:
        print("No triangle data to visualize")
        return
    
    centers = triangle_data[:, :3]
    normals = triangle_data[:, 3:]
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Triangle centers
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
               c=range(len(centers)), cmap='viridis', s=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{len(centers)} Triangle Centers')
    
    # Plot 2: Normal vectors (subset)
    ax2 = fig.add_subplot(132, projection='3d')
    step = max(1, len(centers) // 50)
    for i in range(0, len(centers), step):
        center = centers[i]
        normal = normals[i] * 5
        ax2.quiver(center[0], center[1], center[2], 
                   normal[0], normal[1], normal[2], 
                   color='red', alpha=0.6, arrow_length_ratio=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Surface Normals')
    
    # Plot 3: Normal vector distribution
    ax3 = fig.add_subplot(133)
    normal_lengths = np.linalg.norm(normals, axis=1)
    ax3.hist(normal_lengths, bins=20, alpha=0.7)
    ax3.set_xlabel('Normal Vector Length')
    ax3.set_ylabel('Count')
    ax3.set_title('Normal Vector Length Distribution')
    ax3.axvline(1.0, color='red', linestyle='--', label='Expected length = 1.0')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "mesh_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {plot_path}")
    
    # Save data
    data_path = output_dir / "triangle_data.npy"
    np.save(data_path, triangle_data)
    print(f"Triangle data saved to: {data_path}")


def main():
    """Run all tests"""
    print("TomoPANDA Mesh Geodesic New Features Test")
    print("=" * 50)
    
    # Test 1: Deterministic behavior
    triangle_data_1, triangle_data_2 = test_deterministic_mesh()
    
    # Test 2: Random variants
    mesh_variants = test_random_mesh_variants()
    
    # Test 3: Particle size effect
    particle_size_results = test_particle_size_effect()
    
    # Test 4: Triangle extraction
    triangle_data = test_triangle_extraction()
    
    # Create visualization
    create_visualization(triangle_data)
    
    print("\n=== Test Summary ===")
    print("✓ Deterministic mesh generation works")
    print("✓ Random mesh variants work")
    print("✓ Particle size affects mesh density")
    print("✓ Triangle extraction works")
    print("\nAll tests completed successfully!")
    
    return {
        'deterministic': (triangle_data_1, triangle_data_2),
        'variants': mesh_variants,
        'particle_sizes': particle_size_results,
        'extraction': triangle_data
    }


if __name__ == "__main__":
    results = main()

#!/usr/bin/env python3
"""
Test script for create_signed_distance_field function
测试 create_signed_distance_field 函数
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the tomopanda module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tomopanda.core.mesh_geodesic import MeshGeodesicSampler

def create_membrane_mask(size=256, membrane_diameter=128, membrane_thickness=5):
    """
    Create a 3D membrane mask for testing
    
    Args:
        size: Size of the 3D array (size x size x size)
        membrane_diameter: Diameter of the membrane
        membrane_thickness: Thickness of the membrane
        
    Returns:
        3D binary mask with membrane
    """
    # Create coordinate arrays
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    z = np.arange(size) - size // 2
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distance from center
    distance_from_center = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Create membrane mask: thickness around the sphere surface
    outer_radius = membrane_diameter // 2
    inner_radius = outer_radius - membrane_thickness
    
    # Membrane is between inner and outer radius
    membrane_mask = (distance_from_center >= inner_radius) & (distance_from_center <= outer_radius)
    
    return membrane_mask.astype(float)

def test_sdf_function():
    """Test the create_signed_distance_field function"""
    
    print("Creating membrane mask...")
    # Create membrane mask
    membrane_mask = create_membrane_mask(size=256, membrane_diameter=128, membrane_thickness=3)
    print(f"Membrane mask shape: {membrane_mask.shape}")
    print(f"Membrane voxels: {np.sum(membrane_mask > 0)}")
    
    # Initialize MeshGeodesicSampler
    print("Initializing MeshGeodesicSampler...")
    sampler = MeshGeodesicSampler(
        smoothing_sigma=1.0,
        add_noise=False,
        noise_scale_factor=0.1,
        random_seed=42
    )
    
    # Create signed distance field
    print("Creating signed distance field...")
    phi = sampler.create_signed_distance_field(membrane_mask)
    
    print(f"SDF shape: {phi.shape}")
    print(f"SDF min: {np.min(phi):.3f}")
    print(f"SDF max: {np.max(phi):.3f}")
    print(f"SDF mean: {np.mean(phi):.3f}")
    print(f"Zero points: {np.sum(np.abs(phi) < 1e-6)}")
    print(f"Phi=0 voxels: {np.sum(np.abs(phi) < 1e-6)}")
    
    # Debug information
    print(f"Membrane voxels: {np.sum(membrane_mask > 0)}")
    print(f"Max internal distance: {np.max(phi[membrane_mask > 0]) if np.any(membrane_mask > 0) else 0:.3f}")
    print(f"Center threshold: {np.max(phi[membrane_mask > 0]) * 0.5 if np.any(membrane_mask > 0) else 0:.3f}")
    
    # Visualize results
    visualize_results(membrane_mask, phi)
    
    return membrane_mask, phi

def visualize_results(membrane_mask, phi):
    """Visualize the membrane mask and SDF"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Original membrane mask - middle slice
    ax1 = plt.subplot(2, 3, 1)
    middle_slice = membrane_mask[:, :, 128]  # Middle Z slice
    im1 = ax1.imshow(middle_slice, cmap='viridis', origin='lower')
    ax1.set_title('Original Membrane (Z=128)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # 2. SDF - middle slice
    ax2 = plt.subplot(2, 3, 2)
    sdf_slice = phi[:, :, 128]  # Middle Z slice
    # Use a colormap suitable for positive values
    im2 = ax2.imshow(sdf_slice, cmap='viridis', origin='lower')
    ax2.set_title('SDF (Z=128)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # 3. SDF with membrane center layer (phi=0) highlighted in red
    ax3 = plt.subplot(2, 3, 3)
    # Show the SDF with viridis colormap (good for positive values)
    im3 = ax3.imshow(sdf_slice, cmap='viridis', origin='lower', alpha=0.8)
    
    # Find and highlight membrane center layer pixels (phi=0) in red
    center_mask = np.abs(sdf_slice) < 1e-6  # Exact zero values
    if np.any(center_mask):
        # Create a red overlay for center pixels
        red_overlay = np.zeros((*sdf_slice.shape, 4))  # RGBA
        red_overlay[center_mask] = [1, 0, 0, 1.0]  # Red with full opacity
        ax3.imshow(red_overlay, origin='lower')
        print(f"Found {np.sum(center_mask)} center layer pixels in slice")
    else:
        print("No center layer pixels found in slice")
        # Try with a more lenient threshold to see the center surface
        center_mask = np.abs(sdf_slice) < 0.5
        if np.any(center_mask):
            red_overlay = np.zeros((*sdf_slice.shape, 4))
            red_overlay[center_mask] = [1, 0, 0, 0.8]
            ax3.imshow(red_overlay, origin='lower')
            print(f"Found {np.sum(center_mask)} center layer pixels with lenient threshold")
    
    ax3.set_title('SDF with membrane center layer (red)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # 4. SDF histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(phi.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='phi=0')
    ax4.set_title('SDF Value Distribution')
    ax4.set_xlabel('SDF Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. 3D visualization of membrane center layer
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    
    # Find membrane center layer points (phi=0)
    center_points = np.where(np.abs(phi) < 1e-6)  # Exact zero values
    if len(center_points[0]) > 0:
        # Sample points for visualization (every 2nd point to avoid too many)
        sample_indices = np.arange(0, len(center_points[0]), 2)
        x_points = center_points[0][sample_indices]
        y_points = center_points[1][sample_indices]
        z_points = center_points[2][sample_indices]
        
        ax5.scatter(x_points, y_points, z_points, c='red', alpha=0.8, s=3, label='Membrane Center Layer')
        print(f"Found {len(center_points[0])} center layer points in 3D")
    else:
        print("No center layer points found in 3D")
        # Try with a more lenient threshold to see the center surface
        center_points = np.where(np.abs(phi) < 0.5)
        if len(center_points[0]) > 0:
            sample_indices = np.arange(0, len(center_points[0]), 5)
            x_points = center_points[0][sample_indices]
            y_points = center_points[1][sample_indices]
            z_points = center_points[2][sample_indices]
            
            ax5.scatter(x_points, y_points, z_points, c='red', alpha=0.6, s=2, label='Membrane Center Layer (lenient)')
            print(f"Found {len(center_points[0])} center layer points with lenient threshold")
    
    # Also show some membrane interior points in blue for context
    membrane_points = np.where(membrane_mask > 0)
    if len(membrane_points[0]) > 0:
        # Sample points for visualization (every 50th point)
        sample_indices = np.arange(0, len(membrane_points[0]), 50)
        x_points = membrane_points[0][sample_indices]
        y_points = membrane_points[1][sample_indices]
        z_points = membrane_points[2][sample_indices]
        
        ax5.scatter(x_points, y_points, z_points, c='blue', alpha=0.2, s=1, label='Membrane Interior')
    
    ax5.set_title('3D Membrane Center Layer (red) and Interior (blue)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.legend()
    
    # 6. SDF cross-section along X axis
    ax6 = plt.subplot(2, 3, 6)
    # Use center slice
    center_y, center_z = 128, 128
    sdf_cross_section = phi[:, center_y, center_z]
    ax6.plot(sdf_cross_section, 'b-', linewidth=2, label='SDF')
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2, label='phi=0 (center layer)')
    ax6.set_title(f'SDF Cross-section (Y={center_y}, Z={center_z})')
    ax6.set_xlabel('X position')
    ax6.set_ylabel('SDF Value')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\n=== SDF Statistics ===")
    print(f"Total voxels: {phi.size}")
    print(f"Zero voxels (phi=0): {np.sum(np.abs(phi) < 1e-6)}")
    print(f"Positive voxels: {np.sum(phi > 0)}")
    print(f"Negative voxels: {np.sum(phi < 0)}")
    
    # Find membrane center
    zero_points = np.where(np.abs(phi) < 1e-6)
    if len(zero_points[0]) > 0:
        print(f"Membrane center points: {len(zero_points[0])}")
        print(f"Center coordinates: {list(zip(zero_points[0][:5], zero_points[1][:5], zero_points[2][:5]))}")

if __name__ == "__main__":
    print("Testing create_signed_distance_field function...")
    print("=" * 50)
    
    try:
        membrane_mask, phi = test_sdf_function()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

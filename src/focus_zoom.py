#!/usr/bin/env python3
"""
Create zoomed-in visualizations of the pressure field around the target focal point.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from focus import (
    domain, N, dx_mm, GRID_SIZE_MM, FOCAL_DISTANCE_MM, LENS_START_Z_MM, 
    LENS_THICKNESS_MM, BOWL_ROC_MM, create_bowl_source, compute_field
)

def create_zoom_visualization():
    """Create a zoomed view of the pressure field near the target."""
    
    # Create the bowl source
    bowl_source = create_bowl_source(domain)
    
    # Load parameters
    initial_params = jnp.zeros(60)
    try:
        optimized_params = jnp.array(np.load('/workspace/hologram/outs/optimized_params.npy'))
        print(f"Loaded optimized parameters")
    except FileNotFoundError:
        print("Using approximate optimized parameters")
        optimized_params = jnp.ones(60) * 2.0
    
    # Compute fields
    print("Computing pressure fields...")
    initial_field = compute_field(initial_params, bowl_source, domain)
    optimized_field = compute_field(optimized_params, bowl_source, domain)
    
    # Extract pressure magnitudes
    initial_pressure = jnp.abs(initial_field.on_grid[..., 0])
    optimized_pressure = jnp.abs(optimized_field.on_grid[..., 0])
    
    # Define zoom region around target (in mm)
    zoom_width_mm = 10.0
    zoom_height_mm = 10.0
    target_x_mm = GRID_SIZE_MM[0] / 2
    target_z_mm = FOCAL_DISTANCE_MM
    
    # Convert to voxel indices
    x_center_vox = N[0] // 2
    z_center_vox = int(FOCAL_DISTANCE_MM / dx_mm)
    half_width_vox = int(zoom_width_mm / dx_mm / 2)
    half_height_vox = int(zoom_height_mm / dx_mm / 2)
    
    # Extract zoom regions
    x_start = max(0, x_center_vox - half_width_vox)
    x_end = min(N[0], x_center_vox + half_width_vox)
    z_start = max(0, z_center_vox - half_height_vox)
    z_end = min(N[1], z_center_vox + half_height_vox)
    
    initial_zoom = initial_pressure[x_start:x_end, z_start:z_end]
    optimized_zoom = optimized_pressure[x_start:x_end, z_start:z_end]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full field views (top row)
    vmax_full = 0.1  # Adjust for visibility
    
    # Initial full field
    im1 = axes[0, 0].imshow(initial_pressure.T, 
                           extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                           aspect='auto', cmap='hot', origin='upper',
                           vmin=0, vmax=vmax_full)
    axes[0, 0].set_title('Initial Pressure Field (Full View)')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Z (mm)')
    axes[0, 0].plot(target_x_mm, target_z_mm, 'b*', markersize=15)
    
    # Draw zoom box
    zoom_rect = Rectangle((target_x_mm - zoom_width_mm/2, target_z_mm - zoom_height_mm/2),
                         zoom_width_mm, zoom_height_mm,
                         linewidth=2, edgecolor='cyan', facecolor='none')
    axes[0, 0].add_patch(zoom_rect)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Optimized full field
    im2 = axes[0, 1].imshow(optimized_pressure.T, 
                           extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                           aspect='auto', cmap='hot', origin='upper',
                           vmin=0, vmax=vmax_full)
    axes[0, 1].set_title('Optimized Pressure Field (Full View)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Z (mm)')
    axes[0, 1].plot(target_x_mm, target_z_mm, 'b*', markersize=15)
    
    # Draw zoom box
    zoom_rect2 = Rectangle((target_x_mm - zoom_width_mm/2, target_z_mm - zoom_height_mm/2),
                          zoom_width_mm, zoom_height_mm,
                          linewidth=2, edgecolor='cyan', facecolor='none')
    axes[0, 1].add_patch(zoom_rect2)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference full field
    diff_full = optimized_pressure - initial_pressure
    vmax_diff = jnp.max(jnp.abs(diff_full))
    im3 = axes[0, 2].imshow(diff_full.T, 
                           extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                           aspect='auto', cmap='RdBu_r', origin='upper',
                           vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, 2].set_title('Pressure Difference (Optimized - Initial)')
    axes[0, 2].set_xlabel('X (mm)')
    axes[0, 2].set_ylabel('Z (mm)')
    axes[0, 2].plot(target_x_mm, target_z_mm, 'b*', markersize=15)
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Zoomed views (bottom row)
    zoom_extent = [
        target_x_mm - zoom_width_mm/2,
        target_x_mm + zoom_width_mm/2,
        target_z_mm + zoom_height_mm/2,
        target_z_mm - zoom_height_mm/2
    ]
    
    # Find appropriate scale for zoom views
    vmax_zoom = max(jnp.max(initial_zoom), jnp.max(optimized_zoom))
    
    # Initial zoom
    im4 = axes[1, 0].imshow(initial_zoom.T, 
                           extent=zoom_extent,
                           aspect='auto', cmap='hot', origin='upper',
                           vmin=0, vmax=vmax_zoom)
    axes[1, 0].set_title('Initial (Zoomed)')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Z (mm)')
    axes[1, 0].plot(target_x_mm, target_z_mm, 'b*', markersize=20)
    
    # Add circle to show focal spot size (1 wavelength diameter)
    wavelength_mm = dx_mm * 3  # PPW = 3
    focal_circle = Circle((target_x_mm, target_z_mm), wavelength_mm/2,
                         fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    axes[1, 0].add_patch(focal_circle)
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Optimized zoom
    im5 = axes[1, 1].imshow(optimized_zoom.T, 
                           extent=zoom_extent,
                           aspect='auto', cmap='hot', origin='upper',
                           vmin=0, vmax=vmax_zoom)
    axes[1, 1].set_title('Optimized (Zoomed)')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Z (mm)')
    axes[1, 1].plot(target_x_mm, target_z_mm, 'b*', markersize=20)
    
    focal_circle2 = Circle((target_x_mm, target_z_mm), wavelength_mm/2,
                          fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    axes[1, 1].add_patch(focal_circle2)
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Difference zoom
    diff_zoom = optimized_zoom - initial_zoom
    vmax_diff_zoom = jnp.max(jnp.abs(diff_zoom))
    im6 = axes[1, 2].imshow(diff_zoom.T, 
                           extent=zoom_extent,
                           aspect='auto', cmap='RdBu_r', origin='upper',
                           vmin=-vmax_diff_zoom, vmax=vmax_diff_zoom)
    axes[1, 2].set_title('Difference (Zoomed)')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Z (mm)')
    axes[1, 2].plot(target_x_mm, target_z_mm, 'b*', markersize=20)
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle('Acoustic Focusing Optimization: Full and Zoomed Views', fontsize=16)
    plt.tight_layout()
    plt.savefig('/workspace/hologram/outs/focus_zoom_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print focus quality metrics
    print(f"\nFocus Quality Metrics:")
    print(f"Target position: ({target_x_mm:.1f}, {target_z_mm:.1f}) mm")
    print(f"Wavelength: {wavelength_mm:.3f} mm")
    
    # Calculate focal spot size (FWHM)
    center_idx = (initial_zoom.shape[0]//2, initial_zoom.shape[1]//2)
    
    initial_max = jnp.max(initial_zoom)
    optimized_max = jnp.max(optimized_zoom)
    
    print(f"\nPeak pressure:")
    print(f"  Initial: {initial_max:.4f}")
    print(f"  Optimized: {optimized_max:.4f}")
    print(f"  Improvement: {optimized_max/initial_max:.1f}x")
    
    print(f"\nVisualization saved to: /workspace/hologram/outs/focus_zoom_comparison.png")

if __name__ == "__main__":
    create_zoom_visualization() 
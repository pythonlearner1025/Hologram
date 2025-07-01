#!/usr/bin/env python3
"""
Analyze the focusing optimization results by comparing pressure profiles.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from focus import (
    domain, N, dx_mm, GRID_SIZE_MM, FOCAL_DISTANCE_MM, LENS_START_Z_MM, 
    LENS_THICKNESS_MM, BOWL_ROC_MM, create_bowl_source, compute_field,
    create_optimizable_sos_map
)

# Activate the virtual environment before running this script:
# source /workspace/hologram/jaxenv/bin/activate

def analyze_focusing():
    """Compare initial and optimized focusing along the central axis."""
    
    # Create the bowl source
    bowl_source = create_bowl_source(domain)
    
    # Initial parameters (all zeros)
    initial_params = jnp.zeros(45)  # 60 parameters as shown in the output
    
    # Load the optimized parameters from the saved file
    try:
        optimized_params = jnp.array(np.load('/workspace/hologram/outs/optimized_params.npy'))
        print(f"Loaded optimized parameters from file (shape: {optimized_params.shape})")
    except FileNotFoundError:
        print("Warning: Could not load optimized parameters, using approximate values")
        optimized_params = jnp.ones(45) * 2.0  # Fallback to approximate values
    
    # Compute fields
    print("Computing initial field...")
    initial_field = compute_field(initial_params, bowl_source, domain)
    
    print("Computing optimized field...")
    optimized_field = compute_field(optimized_params, bowl_source, domain)
    
    # Extract pressure magnitudes
    initial_pressure = jnp.abs(initial_field.on_grid[..., 0])
    optimized_pressure = jnp.abs(optimized_field.on_grid[..., 0])
    
    # Get pressure along central axis (x = center)
    center_x = N[0] // 2
    initial_axis = initial_pressure[center_x, :]
    optimized_axis = optimized_pressure[center_x, :]
    
    # Create z-axis in mm
    z_mm = jnp.arange(N[1]) * dx_mm
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Top plot: Pressure profiles along central axis
    plt.subplot(211)
    plt.plot(z_mm, initial_axis, 'b-', label='Initial (no lens optimization)', linewidth=2)
    plt.plot(z_mm, optimized_axis, 'r-', label='Optimized lens', linewidth=2)
    
    # Mark key positions
    plt.axvline(LENS_START_Z_MM, color='gray', linestyle='--', alpha=0.5, label='Lens region')
    plt.axvline(LENS_START_Z_MM + LENS_THICKNESS_MM, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(BOWL_ROC_MM, color='green', linestyle=':', alpha=0.7, label=f'Natural focus (z={BOWL_ROC_MM}mm)')
    plt.axvline(FOCAL_DISTANCE_MM, color='blue', linestyle=':', alpha=0.7, label=f'Target focus (z={FOCAL_DISTANCE_MM}mm)')
    
    plt.xlabel('Z position (mm)')
    plt.ylabel('Pressure Magnitude')
    plt.title('Acoustic Pressure Along Central Axis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom plot: Pressure gain
    plt.subplot(212)
    gain = optimized_axis / (initial_axis + 1e-10)  # Avoid division by zero
    plt.plot(z_mm, gain, 'g-', linewidth=2)
    plt.axhline(1.0, color='black', linestyle='-', alpha=0.3)
    
    # Mark key positions again
    plt.axvline(LENS_START_Z_MM, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(LENS_START_Z_MM + LENS_THICKNESS_MM, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(BOWL_ROC_MM, color='green', linestyle=':', alpha=0.7)
    plt.axvline(FOCAL_DISTANCE_MM, color='blue', linestyle=':', alpha=0.7)
    
    plt.xlabel('Z position (mm)')
    plt.ylabel('Pressure Gain (Optimized/Initial)')
    plt.title('Focusing Improvement Factor')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('/workspace/hologram/outs/focusing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    target_z_idx = int(FOCAL_DISTANCE_MM / dx_mm)
    natural_z_idx = int(BOWL_ROC_MM / dx_mm)
    
    print(f"\nPressure at target focus (z={FOCAL_DISTANCE_MM}mm):")
    print(f"  Initial: {initial_axis[target_z_idx]:.4f}")
    print(f"  Optimized: {optimized_axis[target_z_idx]:.4f}")
    print(f"  Gain: {optimized_axis[target_z_idx]/initial_axis[target_z_idx]:.2f}x")
    
    print(f"\nPressure at natural focus (z={BOWL_ROC_MM}mm):")
    print(f"  Initial: {initial_axis[natural_z_idx]:.4f}")
    print(f"  Optimized: {optimized_axis[natural_z_idx]:.4f}")
    
    # Create 2D comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial pressure field
    im1 = ax1.imshow(initial_pressure.T, 
                     extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                     aspect='auto', cmap='hot', origin='upper',
                     vmin=0, vmax=0.05)  # Use lower vmax to see the focusing
    ax1.set_title('Initial Pressure Field')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'b*', markersize=10)
    plt.colorbar(im1, ax=ax1)
    
    # Optimized pressure field
    im2 = ax2.imshow(optimized_pressure.T, 
                     extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                     aspect='auto', cmap='hot', origin='upper',
                     vmin=0, vmax=0.05)
    ax2.set_title('Optimized Pressure Field')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'b*', markersize=10)
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    im3 = ax3.imshow((optimized_pressure - initial_pressure).T, 
                     extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                     aspect='auto', cmap='RdBu_r', origin='upper',
                     vmin=-0.03, vmax=0.03)
    ax3.set_title('Pressure Difference (Optimized - Initial)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'b*', markersize=10)
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('/workspace/hologram/outs/pressure_comparison_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAnalysis plots saved to:")
    print("  - /workspace/hologram/outs/focusing_analysis.png")
    print("  - /workspace/hologram/outs/pressure_comparison_2d.png")

if __name__ == "__main__":
    analyze_focusing() 
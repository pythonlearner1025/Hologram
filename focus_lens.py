#!/usr/bin/env python3
"""
Stacked lens optimization for 3D-printable acoustic focusing devices.

This implementation optimizes multiple stacked lenses with binary material distribution
(silicone/water) for Formlabs 3D printing with 0.05mm resolution constraint.

Lens structure:
- Each lens: backing layer (1mm) + optimizable layer (0.2mm)
- Gap between lenses: 0.05mm
- X-resolution: 0.05mm (Formlabs constraint)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
import time
import os

from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# Set random seed
key = random.PRNGKey(42)

# Create output directory
OUTPUT_DIR = f'/workspace/hologram/outs/stacked_{time.strftime("%Y%m%d_%H%M%S")}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================

# Acoustic parameters
FREQ_HZ = 1.5e6  # 1.5 MHz
SOUND_SPEED_WATER = 1500  # m/s
DENSITY_WATER = 1000  # kg/m³
ABSORPTION_WATER = 0.0  # Np/m

# Silicone properties
SOUND_SPEED_SILICONE = 1300  # m/s
DENSITY_SILICONE = 1020  # kg/m³
ABSORPTION_SILICONE = 10.0  # Np/m

# Derived parameters
WAVELENGTH = SOUND_SPEED_WATER / FREQ_HZ  # meters
OMEGA = 2 * jnp.pi * FREQ_HZ

# Bowl transducer
BOWL_DIAMETER_MM = 15.0
BOWL_ROC_MM = 23.0  # Radius of curvature

# Target
FOCAL_DISTANCE_MM = 25.0
TARGET_RADIUS_MM = 1.0

# ============================================================================
# LENS CONFIGURATION
# ============================================================================

# Target area parameters (for area-based objectives)
SIDELOBE_RADIUS_MM = 8.0  # Radius for sidelobe suppression region

# Lens stack parameters
NUM_LENSES = 3
LENS_WIDTH_MM = 15.0
LENS_THICKNESS_MM = 0.5  # Optimizable layer thickness
BACKING_THICKNESS_MM = 1.0  # Non-optimizable backing
GAP_BETWEEN_LENSES_MM = 0.1
LENS_START_Z_MM = 3.0  # Where first lens begins

# Formlabs resolution constraint
MAX_X_RESOLUTION_MM = 0.05  # Minimum feature size

# Calculate number of controllable segments per lens
NUM_X_SEGMENTS = int(LENS_WIDTH_MM / MAX_X_RESOLUTION_MM)  # 300 segments

# ============================================================================
# GRID CONFIGURATION
# ============================================================================

# Grid size
GRID_SIZE_MM = (20, 50)  # x, z dimensions

# Choose grid spacing
# Option 1: Match Formlabs resolution
# dx_mm = MAX_X_RESOLUTION_MM  # 0.05mm - very fine, slow
# Option 2: Use reasonable multiple for faster simulation
dx_mm = 0.1  # 2x Formlabs resolution
dx_m = dx_mm / 1000

# ============================================================================
# Visualization Configuration 
# ============================================================================

VMAX = 0.5

# Grid dimensions in voxels
N = (
    int(GRID_SIZE_MM[0] / dx_mm),
    int(GRID_SIZE_MM[1] / dx_mm)
)

# Create domain
domain = Domain(N, (dx_m, dx_m))

print(f"Grid configuration:")
print(f"  Grid size: {N} voxels")
print(f"  Grid spacing: {dx_mm} mm")
print(f"  Domain size: {GRID_SIZE_MM[0]} x {GRID_SIZE_MM[1]} mm")
print(f"\nLens configuration:")
print(f"  Number of lenses: {NUM_LENSES}")
print(f"  Segments per lens: {NUM_X_SEGMENTS}")
print(f"  Total parameters: {NUM_LENSES * NUM_X_SEGMENTS}")

# ============================================================================
# SOURCE DEFINITION
# ============================================================================

def create_bowl_source(domain):
    """Create a bowl-shaped transducer source."""
    # Define grid coordinates in mm
    x = jnp.arange(N[0]) * dx_mm - GRID_SIZE_MM[0] / 2
    z = jnp.arange(N[1]) * dx_mm
    X, Z = jnp.meshgrid(x, z, indexing='ij')
    
    # Bowl geometry calculations
    bowl_half_diameter = BOWL_DIAMETER_MM / 2
    # Place the deepest part of the bowl at z=0
    bowl_center_z = BOWL_ROC_MM
    
    # Distance from the bowl's center of curvature
    R = jnp.sqrt(X**2 + (Z - bowl_center_z)**2)
    
    # Create a mask to define the transducer surface
    on_surface = jnp.abs(R - BOWL_ROC_MM) < dx_mm
    within_aperture = jnp.abs(X) <= bowl_half_diameter
    
    # Add rectangular mask to chunk out only the bowl portion
    # Convert the specified coordinates to the centered coordinate system
    left_bottom_x_mm = (GRID_SIZE_MM[0] - BOWL_DIAMETER_MM) / 2  # 5 mm in absolute coords
    right_top_x_mm = GRID_SIZE_MM[0] - (GRID_SIZE_MM[0] - BOWL_DIAMETER_MM) / 2  # 20 mm in absolute coords
    left_bottom_z_mm = 0
    right_top_z_mm = GRID_SIZE_MM[1] / 2  # 25 mm
    
    # Convert to centered x coordinates
    left_x_centered = left_bottom_x_mm - GRID_SIZE_MM[0] / 2  # -7.5 mm
    right_x_centered = right_top_x_mm - GRID_SIZE_MM[0] / 2   # 7.5 mm
    
    # Create rectangular mask
    rect_mask = (X >= left_x_centered) & (X <= right_x_centered) & \
                (Z >= left_bottom_z_mm) & (Z <= right_top_z_mm)
    
    # Combine all masks: on surface, within aperture, AND within rectangle
    bowl_mask = on_surface & within_aperture & rect_mask
    
    # Create source field on the grid
    # Phase is applied to pre-focus the wave towards the radius of curvature center
    phase = -OMEGA / SOUND_SPEED_WATER * R
    source_field = jnp.where(bowl_mask, 1.0 * jnp.exp(1j * phase), 0.0+0j)
    
    return FourierSeries(jnp.expand_dims(source_field, -1), domain)
# ============================================================================
# STACKED LENS MODEL
# ============================================================================

def create_stacked_lenses_sos_map(lens_params, domain):
    """
    Create sound speed map for stacked lenses with binary material optimization.
    
    Args:
        lens_params: Array of size (NUM_LENSES * NUM_X_SEGMENTS)
                    Each lens has NUM_X_SEGMENTS parameters
    """
    # Check parameter count
    expected_params = NUM_LENSES * NUM_X_SEGMENTS
    if len(lens_params) != expected_params:
        raise ValueError(f"Expected {expected_params} parameters, got {len(lens_params)}")
    
    # Initialize with water
    sos_map = jnp.ones(N) * SOUND_SPEED_WATER
    
    # Get lens region bounds in x
    lens_x_start_mm = (GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2
    lens_x_start_vox = int(lens_x_start_mm / dx_mm)
    
    # Process each lens
    current_z_mm = LENS_START_Z_MM
    
    for lens_idx in range(NUM_LENSES):
        # Extract parameters for this lens
        start_idx = lens_idx * NUM_X_SEGMENTS
        end_idx = (lens_idx + 1) * NUM_X_SEGMENTS
        lens_i_params = lens_params[start_idx:end_idx]
        
        # Apply sigmoid to get material fractions (0=water, 1=silicone)
        material_fractions = jax.nn.sigmoid(lens_i_params)
        
        # Convert to voxel coordinates
        lens_z_start_vox = int(current_z_mm / dx_mm)
        lens_thickness_vox = int(LENS_THICKNESS_MM / dx_mm)
        backing_thickness_vox = int(BACKING_THICKNESS_MM / dx_mm)
        
        # Fill optimizable layer
        # Each parameter controls a MAX_X_RESOLUTION_MM wide segment
        for seg_idx in range(NUM_X_SEGMENTS):
            # Segment bounds in mm
            seg_x_start_mm = seg_idx * MAX_X_RESOLUTION_MM
            seg_x_end_mm = (seg_idx + 1) * MAX_X_RESOLUTION_MM
            
            # Convert to voxel coordinates (relative to lens start)
            seg_x_start_vox = int(seg_x_start_mm / dx_mm)
            seg_x_end_vox = int(seg_x_end_mm / dx_mm)
            
            # Calculate effective sound speed for this segment
            seg_sos = (SOUND_SPEED_SILICONE - SOUND_SPEED_WATER) * material_fractions[seg_idx] + SOUND_SPEED_WATER
            
            # Apply to map
            x_start = lens_x_start_vox + seg_x_start_vox
            x_end = lens_x_start_vox + seg_x_end_vox
            z_start = lens_z_start_vox
            z_end = lens_z_start_vox + lens_thickness_vox
            
            sos_map = sos_map.at[x_start:x_end, z_start:z_end].set(seg_sos)
        
        # Add backing layer (uniform silicone)
        backing_z_start_vox = lens_z_start_vox + lens_thickness_vox
        backing_z_end_vox = backing_z_start_vox + backing_thickness_vox
        
        sos_map = sos_map.at[
            lens_x_start_vox:lens_x_start_vox + int(LENS_WIDTH_MM / dx_mm),
            backing_z_start_vox:backing_z_end_vox
        ].set(SOUND_SPEED_SILICONE)
        
        # Move to next lens position (add gap)
        current_z_mm += LENS_THICKNESS_MM + BACKING_THICKNESS_MM + GAP_BETWEEN_LENSES_MM
    
    # Print binarization statistics
    nearly_water = jnp.sum(material_fractions < 0.05)
    nearly_silicone = jnp.sum(material_fractions > 0.95)
    total = len(material_fractions)
    binary_percentage = 100.0 * (nearly_water + nearly_silicone) / total
    
    return FourierSeries(jnp.expand_dims(sos_map, -1), domain)

# ============================================================================
# SIMULATION
# ============================================================================

# Helper function for creating target masks
def create_circular_mask(center_x_mm, center_z_mm, radius_mm):
    """Create a circular mask for the target area."""
    x = jnp.arange(N[0]) * dx_mm
    z = jnp.arange(N[1]) * dx_mm
    X, Z = jnp.meshgrid(x, z, indexing='ij')
    
    # Distance from center
    dist = jnp.sqrt((X - center_x_mm)**2 + (Z - center_z_mm)**2)
    
    return dist <= radius_mm

def objective_focal_point(lens_params, source, domain):
    """Maximize pressure at focal point."""
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    # Target location
    target_x_vox = N[0] // 2
    target_z_vox = int(FOCAL_DISTANCE_MM / dx_mm)
    
    target_pressure = pressure_magnitude[target_x_vox, target_z_vox]
    
    return -target_pressure  # Minimize negative pressure

def objective_focal_area(lens_params, source, domain):
    """Maximize average pressure in focal area."""
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    # Create circular target mask
    x = jnp.arange(N[0]) * dx_mm
    z = jnp.arange(N[1]) * dx_mm
    X, Z = jnp.meshgrid(x, z, indexing='ij')
    
    center_x_mm = GRID_SIZE_MM[0] / 2
    center_z_mm = FOCAL_DISTANCE_MM
    
    dist = jnp.sqrt((X - center_x_mm)**2 + (Z - center_z_mm)**2)
    target_mask = (dist <= TARGET_RADIUS_MM).astype(jnp.float32)
    
    # Calculate mean pressure in target area
    target_sum = jnp.sum(pressure_magnitude * target_mask)
    target_count = jnp.sum(target_mask)
    mean_pressure = target_sum / (target_count + 1e-8)
    
    return -mean_pressure
    
# Area-based objective functions
def objective_area_contrast(lens_params, source, domain):
    """
    Maximize contrast between target area and surrounding region.
    Contrast = (mean pressure in target) / (mean pressure outside target)
    """
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    # Create target area mask
    target_mask = create_circular_mask(
        center_x_mm=GRID_SIZE_MM[0]/2,
        center_z_mm=FOCAL_DISTANCE_MM,
        radius_mm=TARGET_RADIUS_MM
    )
    
    # Convert boolean mask to float for JAX compatibility
    target_mask_float = target_mask.astype(jnp.float32)
    non_target_mask_float = 1.0 - target_mask_float
    
    # Calculate mean pressures using weighted sums
    target_sum = jnp.sum(pressure_magnitude * target_mask_float)
    target_count = jnp.sum(target_mask_float)
    target_pressure = target_sum / (target_count + 1e-8)
    
    non_target_sum = jnp.sum(pressure_magnitude * non_target_mask_float)
    non_target_count = jnp.sum(non_target_mask_float)
    non_target_pressure = non_target_sum / (non_target_count + 1e-8)
    
    # Maximize contrast ratio
    contrast = target_pressure / (non_target_pressure + 1e-8)
    
    return -contrast  # Minimize negative contrast


def objective_uniform(lens_params, source, domain):
    """
    Create uniform pressure distribution within target area.
    Loss = -(mean_pressure - 0.1 * pressure_variance)
    """
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    # Create target area mask
    target_mask = create_circular_mask(
        center_x_mm=GRID_SIZE_MM[0]/2,
        center_z_mm=FOCAL_DISTANCE_MM,
        radius_mm=TARGET_RADIUS_MM
    )
    
    # Convert to float for JAX
    target_mask_float = target_mask.astype(jnp.float32)
    
    # Calculate mean using weighted sum
    target_sum = jnp.sum(pressure_magnitude * target_mask_float)
    target_count = jnp.sum(target_mask_float)
    mean_pressure = target_sum / (target_count + 1e-8)
    
    # Calculate variance
    squared_diff = (pressure_magnitude - mean_pressure) ** 2
    variance_sum = jnp.sum(squared_diff * target_mask_float)
    pressure_variance = variance_sum / (target_count + 1e-8)
    
    # Maximize mean while minimizing variance
    return -(mean_pressure - 0.1 * pressure_variance)


def objective_sidelobe(lens_params, source, domain):
    """
    Maximize pressure in target area while suppressing sidelobes.
    Loss = -(target_pressure - 0.3 * sidelobe_pressure)
    """
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    # Create masks
    target_mask = create_circular_mask(
        center_x_mm=GRID_SIZE_MM[0]/2,
        center_z_mm=FOCAL_DISTANCE_MM,
        radius_mm=TARGET_RADIUS_MM
    )
    
    sidelobe_region = create_circular_mask(
        center_x_mm=GRID_SIZE_MM[0]/2,
        center_z_mm=FOCAL_DISTANCE_MM,
        radius_mm=SIDELOBE_RADIUS_MM
    )
    
    # Convert to float and create sidelobe mask
    target_mask_float = target_mask.astype(jnp.float32)
    sidelobe_region_float = sidelobe_region.astype(jnp.float32)
    sidelobe_mask_float = sidelobe_region_float - target_mask_float  # Annular region
    
    # Calculate mean pressures
    target_sum = jnp.sum(pressure_magnitude * target_mask_float)
    target_count = jnp.sum(target_mask_float)
    target_pressure = target_sum / (target_count + 1e-8)
    
    sidelobe_sum = jnp.sum(pressure_magnitude * sidelobe_mask_float)
    sidelobe_count = jnp.sum(sidelobe_mask_float)
    sidelobe_pressure = sidelobe_sum / (sidelobe_count + 1e-8)
    
    # Weighted objective
    return -(target_pressure - 0.3 * sidelobe_pressure)

def compute_field(lens_params, source, domain):
    """Compute acoustic field for given lens parameters."""
    sound_speed = create_stacked_lenses_sos_map(lens_params, domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=15)
    
    field = helmholtz_solver(
        medium,
        OMEGA,
        source,
        guess=None,
        tol=1e-4,
        checkpoint=False
    )
    
    return field

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_lens_stack(lens_params, save_path):
    """Visualize the binary material distribution in the lens stack."""
    fig, axes = plt.subplots(NUM_LENSES, 2, figsize=(12, 4*NUM_LENSES))
    if NUM_LENSES == 1:
        axes = axes.reshape(1, -1)
    
    for lens_idx in range(NUM_LENSES):
        # Extract parameters for this lens
        start_idx = lens_idx * NUM_X_SEGMENTS
        end_idx = (lens_idx + 1) * NUM_X_SEGMENTS
        lens_i_params = lens_params[start_idx:end_idx]
        
        # Get material fractions
        material_fractions = jax.nn.sigmoid(lens_i_params)
        
        # Reshape for visualization
        # Each segment is MAX_X_RESOLUTION_MM wide
        fractions_2d = jnp.repeat(material_fractions.reshape(-1, 1), 
                                 int(LENS_THICKNESS_MM / MAX_X_RESOLUTION_MM), 
                                 axis=1).T
        
        # Continuous
        im1 = axes[lens_idx, 0].imshow(fractions_2d, 
                                      extent=[0, LENS_WIDTH_MM, 0, LENS_THICKNESS_MM],
                                      aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
        axes[lens_idx, 0].set_title(f'Lens {lens_idx+1}: Continuous')
        axes[lens_idx, 0].set_xlabel('X (mm)')
        axes[lens_idx, 0].set_ylabel('Z (mm)')
        plt.colorbar(im1, ax=axes[lens_idx, 0])
        
        # Binary
        binary = (material_fractions > 0.5).astype(float)
        binary_2d = jnp.repeat(binary.reshape(-1, 1), 
                              int(LENS_THICKNESS_MM / MAX_X_RESOLUTION_MM), 
                              axis=1).T
        
        im2 = axes[lens_idx, 1].imshow(binary_2d,
                                      extent=[0, LENS_WIDTH_MM, 0, LENS_THICKNESS_MM],
                                      aspect='auto', cmap='binary', vmin=0, vmax=1)
        axes[lens_idx, 1].set_title(f'Lens {lens_idx+1}: Binary')
        axes[lens_idx, 1].set_xlabel('X (mm)')
        axes[lens_idx, 1].set_ylabel('Z (mm)')
        
        # Add grid
        for i in range(0, NUM_X_SEGMENTS+1, 10):
            x = i * MAX_X_RESOLUTION_MM
            axes[lens_idx, 1].axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_sound_speed_map(lens_params, save_path):
    """Visualize the complete sound speed map."""
    sos_map = create_stacked_lenses_sos_map(lens_params, domain)
    sos_on_grid = sos_map.on_grid[..., 0]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(sos_on_grid.T, 
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                   aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(im, label='Sound Speed (m/s)')
    plt.title('Sound Speed Map with Stacked Lenses')
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    
    # Mark lens positions
    lens_x_start = (GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2
    current_z = LENS_START_Z_MM
    
    for i in range(NUM_LENSES):
        # Optimizable layer
        rect1 = plt.Rectangle((lens_x_start, current_z), 
                            LENS_WIDTH_MM, LENS_THICKNESS_MM,
                            linewidth=2, edgecolor='red', facecolor='none',
                            label=f'Lens {i+1}' if i == 0 else '')
        plt.gca().add_patch(rect1)
        
        # Backing
        rect2 = plt.Rectangle((lens_x_start, current_z + LENS_THICKNESS_MM), 
                            LENS_WIDTH_MM, BACKING_THICKNESS_MM,
                            linewidth=2, edgecolor='orange', facecolor='none',
                            linestyle='--',
                            label='Backing' if i == 0 else '')
        plt.gca().add_patch(rect2)
        
        current_z += LENS_THICKNESS_MM + BACKING_THICKNESS_MM + GAP_BETWEEN_LENSES_MM
    
    # Mark focal point
    plt.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'r*', markersize=10, label='Target')
    
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_pressure_field(lens_params, source, save_path):
    """Visualize the acoustic pressure field."""
    field = compute_field(lens_params, source, domain)
    field_on_grid = field.on_grid
    
    if field_on_grid.ndim == 3:
        pressure_magnitude = jnp.abs(field_on_grid[..., 0])
    else:
        pressure_magnitude = jnp.abs(field_on_grid)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(pressure_magnitude.T,
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                   aspect='auto', cmap='hot', origin='upper', vmin=0, vmax=VMAX)
    plt.colorbar(im, label='Pressure Magnitude')
    plt.title('Acoustic Pressure Field')
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    
    # Mark target
    plt.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'b*', markersize=10, label='Target')
    
    # Mark target area
    circle = plt.Circle((GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM), 
                       TARGET_RADIUS_MM, color='cyan', fill=False, 
                       linewidth=2, label='Target Area')
    plt.gca().add_patch(circle)
    
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_binary_design(lens_params, output_dir):
    """Export binary design for 3D printing."""
    import csv
    
    for lens_idx in range(NUM_LENSES):
        # Extract parameters for this lens
        start_idx = lens_idx * NUM_X_SEGMENTS
        end_idx = (lens_idx + 1) * NUM_X_SEGMENTS
        lens_i_params = lens_params[start_idx:end_idx]
        
        # Get binary design
        material_fractions = jax.nn.sigmoid(lens_i_params)
        binary_design = (material_fractions > 0.5).astype(float)
        
        # Export CSV for this lens
        filename = os.path.join(output_dir, f'lens_{lens_idx+1}_design.csv')
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['segment_idx', 'x_start_mm', 'x_end_mm', 'material'])
            
            for seg_idx in range(NUM_X_SEGMENTS):
                x_start = seg_idx * MAX_X_RESOLUTION_MM
                x_end = (seg_idx + 1) * MAX_X_RESOLUTION_MM
                material = 'silicone' if binary_design[seg_idx] > 0.5 else 'void'
                writer.writerow([seg_idx, x_start, x_end, material])
        
        print(f"Exported lens {lens_idx+1} design to {filename}")
# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_lenses():
    """Main optimization routine."""
    print("\n" + "="*60)
    print("STACKED LENS OPTIMIZATION")
    print("="*60)
    
    # Create source
    bowl_source = create_bowl_source(domain)
    
    # Initialize parameters
    total_params = NUM_LENSES * NUM_X_SEGMENTS
    initial_params = 0.2 * (random.uniform(key, (total_params,)) - 0.5)
    
    # Choose objective
    objective_fn = objective_sidelobe  # or objective_focal_point
    print(f"Using objective: {objective_fn.__name__}")
    
    # Set up optimizer
    learning_rate = 0.1
    init_fun, update_fun, get_params = optimizers.adam(learning_rate)
    opt_state = init_fun(initial_params)
    
    # Create gradient function
    objective_and_grad = value_and_grad(objective_fn, argnums=0)
    
    # JIT compile optimization step
    @jit
    def optimization_step(opt_state, source, domain, iteration):
        params = get_params(opt_state)
        loss, grad = objective_and_grad(params, source, domain)
        opt_state = update_fun(iteration, grad, opt_state)
        return opt_state, loss
    
    # Optimization loop
    n_iterations = 50
    print(f"Running {n_iterations} iterations...\n")
    
    start_time = time.time()
    losses = []
    
    for i in range(n_iterations):
        opt_state, loss = optimization_step(opt_state, bowl_source, domain, i)
        losses.append(float(loss))
        
        if i % 10 == 0:
            print(f"Iter {i:3d}: Loss = {loss:.6f}")
            
            # Save intermediate results
            if i > 0:
                params = get_params(opt_state)
                visualize_lens_stack(params, f'{OUTPUT_DIR}/lenses_iter_{i:03d}.png')
                visualize_pressure_field(params, bowl_source, f'{OUTPUT_DIR}/pressure_iter_{i:03d}.png')
    
    # Get final parameters
    final_params = get_params(opt_state)
    elapsed = time.time() - start_time
    
    print(f"\nOptimization complete in {elapsed:.1f}s")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Improvement: {(losses[0] - losses[-1])/abs(losses[0])*100:.1f}%")
    
    # Save results
    print("\nSaving results...")
    visualize_lens_stack(final_params, f'{OUTPUT_DIR}/final_lenses.png')
    visualize_sound_speed_map(final_params, f'{OUTPUT_DIR}/final_sos_map.png')
    visualize_pressure_field(final_params, bowl_source, f'{OUTPUT_DIR}/final_pressure.png')
    export_binary_design(final_params, OUTPUT_DIR)
    
    # Save parameters
    np.save(f'{OUTPUT_DIR}/optimized_params.npy', np.array(final_params))
    
    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimization Convergence')
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/convergence.png', dpi=150)
    plt.close()
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    # Save configuration
    import json
    
    config = {
        # Physical parameters
        'freq_hz': FREQ_HZ,
        'sound_speed_water': SOUND_SPEED_WATER,
        'density_water': DENSITY_WATER,
        'absorption_water': ABSORPTION_WATER,
        'sound_speed_silicone': SOUND_SPEED_SILICONE,
        'density_silicone': DENSITY_SILICONE,
        'absorption_silicone': ABSORPTION_SILICONE,
        'wavelength': float(WAVELENGTH),
        'omega': float(OMEGA),
        
        # Bowl transducer
        'bowl_diameter_mm': BOWL_DIAMETER_MM,
        'bowl_roc_mm': BOWL_ROC_MM,
        
        # Target
        'focal_distance_mm': FOCAL_DISTANCE_MM,
        'target_radius_mm': TARGET_RADIUS_MM,
        'sidelobe_radius_mm': SIDELOBE_RADIUS_MM,
        
        # Lens configuration
        'num_lenses': NUM_LENSES,
        'lens_width_mm': LENS_WIDTH_MM,
        'lens_thickness_mm': LENS_THICKNESS_MM,
        'backing_thickness_mm': BACKING_THICKNESS_MM,
        'gap_between_lenses_mm': GAP_BETWEEN_LENSES_MM,
        'lens_start_z_mm': LENS_START_Z_MM,
        'max_x_resolution_mm': MAX_X_RESOLUTION_MM,
        'num_x_segments': NUM_X_SEGMENTS,
        
        # Grid configuration
        'grid_size_mm': GRID_SIZE_MM,
        'dx_mm': dx_mm,
        'dx_m': dx_m,
        'vmax': VMAX,
        'n': list(N),
        
        # Training results
        'final_loss': float(losses[-1]),
        'initial_loss': float(losses[0]),
        'improvement_percent': float((losses[0] - losses[-1])/abs(losses[0])*100),
        'training_time_sec': elapsed,
        'num_iterations': n_iterations,
        'loss_history': [float(l) for l in losses]
    }
    
    with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    return final_params

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    optimize_lenses()
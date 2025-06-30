#!/usr/bin/env python3
"""
Implementation of bowl transducer with acoustic lens optimization using j-Wave.

This script is adapted from the "Speed of sound gradients" section of the j-Wave
tutorial. It optimizes the sound speed distribution within a fixed region (the lens)
to focus the acoustic field from a bowl transducer onto a target point.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from jax.example_libraries import optimizers
from functools import partial
import matplotlib.pyplot as plt

# Ensure j-wave is installed: pip install j-wave
from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# Set random seed for reproducibility
key = random.PRNGKey(42)

# Timestamp of last modification
import time
import os
TIMESTAMP = time.strftime("%Y-%m-%d %H:%M:%S")  # UTC

# Create output directory with timestamp
OUTPUT_DIR = f'/workspace/hologram/outs/{TIMESTAMP}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. PHYSICAL AND GRID PARAMETERS
# ==================================

# Physical parameters
FREQ_HZ = 1.5e6  # 1.5 MHz
SOUND_SPEED_WATER = 1500  # m/s
WAVELENGTH = SOUND_SPEED_WATER / FREQ_HZ  # meters
OMEGA = 2 * jnp.pi * FREQ_HZ

# Bowl transducer parameters (in mm)
BOWL_DIAMETER_MM = 15.0
BOWL_ROC_MM = 23.0  # Radius of curvature
VMAX = 0.1

# Target focal point
FOCAL_DISTANCE_MM = 30.0 # Target focal point distance from origin (z-axis)

# Objective function type selection
# Options: "point", "area_contrast", "uniform", "sidelobe"
OBJECTIVE_TYPE = "sidelobe"

# Target area parameters (for area-based objectives)
TARGET_RADIUS_MM = 2.0  # Radius of circular target area
SIDELOBE_RADIUS_MM = 5.0  # Radius for sidelobe suppression region

# Grid parameters
GRID_SIZE_MM = (25, 50)  # x, z dimensions in mm
PPW = 3  # Points per wavelength
dx_m = WAVELENGTH / PPW  # Grid spacing in meters
dx_mm = dx_m * 1000  # Grid spacing in mm

# Grid dimensions in voxels
N = (
    int(GRID_SIZE_MM[0] / dx_mm),
    int(GRID_SIZE_MM[1] / dx_mm)
)

# Create simulation domain
domain = Domain(N, (dx_m, dx_m))

# Lens region parameters
# This is the area where we will optimize the sound speed
NUM_LENS_STACKS = 3
LENS_WIDTH_MM = 15.0
LENS_THICKNESS_MM = 3.0
LENS_START_Z_MM = 3.0 # z-position where the lens begins

# 2. SOURCE AND LENS DEFINITIONS
# ===============================

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


def create_optimizable_sos_map(lens_params, domain):
    """
    Creates a sound speed map where a specific region (the lens) has its
    sound speed controlled by the `lens_params` vector.
    This is based on the tutorial's `get_sos` function.
    """
    # Start with a uniform sound speed of water
    sos_map = jnp.ones(N) * SOUND_SPEED_WATER
    
    # Define the grid coordinates for the lens region in voxels
    lens_x_center_vox = N[0] // 2
    lens_width_vox = int(LENS_WIDTH_MM / dx_mm)
    lens_start_x_vox = lens_x_center_vox - lens_width_vox // 2
    
    lens_start_z_vox = int(LENS_START_Z_MM / dx_mm)
    lens_thickness_vox = int(LENS_THICKNESS_MM / dx_mm)
    
    # Ensure the number of lens parameters matches the number of segments
    num_segments = lens_width_vox
    if len(lens_params) != num_segments:
        raise ValueError(f"lens_params length ({len(lens_params)}) must match lens width in voxels ({num_segments})")

    # Use jax.nn.sigmoid to map the parameters to a sound speed multiplier
    # This keeps the sound speed within a reasonable range (e.g., 0.7 to 1.3 times water)
    # The multiplier is shaped to match the segments of the lens
    # TODO the range should be defined by printable material properties
    sound_speed_multiplier = 0.7 + 0.6 * jax.nn.sigmoid(lens_params)
    
    # Create a broadcastable shape for the multiplier
    multiplier_reshaped = jnp.reshape(sound_speed_multiplier, (num_segments, 1))
    # Broadcast to match lens thickness
    multiplier_reshaped = jnp.broadcast_to(multiplier_reshaped, (num_segments, lens_thickness_vox))

    print(f"sound_speed_multiplier: {sound_speed_multiplier.shape}")

    # Apply the multiplier to the lens region in the sound speed map
    # Each parameter controls a vertical slice of the lens
    lens_region = sos_map[lens_start_x_vox : lens_start_x_vox + num_segments,
                          lens_start_z_vox : lens_start_z_vox + lens_thickness_vox]
    
    updated_lens_region = SOUND_SPEED_WATER * multiplier_reshaped
    
    sos_map = sos_map.at[
        lens_start_x_vox : lens_start_x_vox + num_segments,
        lens_start_z_vox : lens_start_z_vox + lens_thickness_vox
    ].set(updated_lens_region)

    return FourierSeries(jnp.expand_dims(sos_map, -1), domain)


# 3. SIMULATION AND OBJECTIVE FUNCTION
# ====================================

def compute_field(lens_params, source, domain):
    """Compute the acoustic field for the given lens sound speed parameters."""
    sound_speed = create_optimizable_sos_map(lens_params, domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=20)
    
    # Based on the tutorial, helmholtz_solver expects:
    # helmholtz_solver(medium, omega, source, guess=None, tol=1e-4, checkpoint=False)
    field = helmholtz_solver(
        medium,
        OMEGA,  # omega directly, not normalized
        source,  # FourierSeries source
        guess=None,
        tol=1e-4,
        checkpoint=False  # Must be False for differentiation
    )
    
    return field

def objective(lens_params, source, domain):
    """
    Objective function: maximize the acoustic pressure at the target point.
    We minimize the negative of the pressure.
    """
    field = compute_field(lens_params, source, domain)
    # Based on the tutorial, field.on_grid gives us the pressure field
    field_on_grid = field.on_grid
    
    # Define target position in grid coordinates (voxels)
    target_x_vox = N[0] // 2
    target_z_vox = int(FOCAL_DISTANCE_MM / dx_mm)
    
    # Extract the pressure at the target point (remove the last dimension if needed)
    if field_on_grid.ndim == 3:
        target_pressure = jnp.abs(field_on_grid[target_x_vox, target_z_vox, 0])
    else:
        target_pressure = jnp.abs(field_on_grid[target_x_vox, target_z_vox])
    
    # The goal is to maximize pressure, so the loss is the negative pressure
    return -target_pressure


# Helper function for creating target masks
def create_circular_mask(center_x_mm, center_z_mm, radius_mm):
    """Create a circular mask for the target area."""
    x = jnp.arange(N[0]) * dx_mm
    z = jnp.arange(N[1]) * dx_mm
    X, Z = jnp.meshgrid(x, z, indexing='ij')
    
    # Distance from center
    dist = jnp.sqrt((X - center_x_mm)**2 + (Z - center_z_mm)**2)
    
    return dist <= radius_mm


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


# 4. OPTIMIZATION
# =================

def main():
    print(f"Grid size: {N} voxels")
    print(f"Grid spacing: {dx_mm:.3f} mm")
    print(f"Target focus: {FOCAL_DISTANCE_MM} mm (z = {int(FOCAL_DISTANCE_MM / dx_mm)} voxels)")
    print(f"Objective type: {OBJECTIVE_TYPE}")
    
    # Create the fixed bowl source
    bowl_source = create_bowl_source(domain)
    
    # Select objective function based on configuration
    if OBJECTIVE_TYPE == "point":
        objective_fn = objective
        print("Using point-based objective")
    elif OBJECTIVE_TYPE == "area_contrast":
        objective_fn = objective_area_contrast
        print(f"Using area contrast objective (target radius: {TARGET_RADIUS_MM} mm)")
    elif OBJECTIVE_TYPE == "uniform":
        objective_fn = objective_uniform
        print(f"Using uniform area objective (target radius: {TARGET_RADIUS_MM} mm)")
    elif OBJECTIVE_TYPE == "sidelobe":
        objective_fn = objective_sidelobe
        print(f"Using sidelobe suppression objective (target: {TARGET_RADIUS_MM} mm, sidelobe: {SIDELOBE_RADIUS_MM} mm)")
    else:
        raise ValueError(f"Unknown objective type: {OBJECTIVE_TYPE}")
    
    # Debug: Visualize the bowl source
    plt.figure(figsize=(10, 6))
    source_magnitude = jnp.abs(bowl_source.on_grid[..., 0])
    im = plt.imshow(source_magnitude.T, 
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                   aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(im, label='Source Magnitude')
    plt.title('Bowl Transducer Source Distribution')
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    
    # Mark the natural focus
    plt.plot(GRID_SIZE_MM[0]/2, BOWL_ROC_MM, 'r+', markersize=10, label=f'Center of Curvature (z={BOWL_ROC_MM}mm)')
    plt.legend()
    
    plt.savefig(f'{OUTPUT_DIR}/bowl_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved bowl source visualization to /workspace/hologram/outs/bowl_source.png")
    
    # Initialize the lens parameters. One parameter per vertical slice of the lens.
    # Three stackable lenses, each with lens_width_vox parameters
    
    lens_width_vox = int(LENS_WIDTH_MM / dx_mm)
    initial_lens_params = jnp.zeros(lens_width_vox * 3)
    
    print(f"Lens region width: {LENS_WIDTH_MM} mm ({lens_width_vox} optimizable parameters)")

    # Set up the Adam optimizer
    learning_rate = 0.1
    init_fun, update_fun, get_params = optimizers.adam(learning_rate)
    opt_state = init_fun(initial_lens_params)
    
    # Create the function that computes both the loss and the gradient
    # The `argnums=0` specifies that we want to differentiate with respect to the first argument (`lens_params`)
    objective_and_grad = value_and_grad(objective_fn, argnums=0)
    
    # JIT-compile the optimization step for speed
    @jit
    def optimization_step(opt_state, source, domain, iteration):
        params = get_params(opt_state)
        loss, grad = objective_and_grad(params, source, domain)
        opt_state = update_fun(iteration, grad, opt_state)
        return opt_state, loss
    
    # Run the optimization loop
    n_iterations = 30
    print("\nStarting optimization...")
    
    import time
    start_time = time.time()
    
    # Track pressure values for debugging
    pressure_history = []
    
    # Create a function to visualize and save the pressure field
    def save_pressure_field(lens_params, iteration, vmin=None, vmax=None):
        """Compute and save the pressure field for the current lens parameters."""
        field = compute_field(lens_params, bowl_source, domain)
        field_on_grid = field.on_grid
        
        # Extract the magnitude of the pressure field
        if field_on_grid.ndim == 3:
            pressure_magnitude = jnp.abs(field_on_grid[..., 0])
        else:
            pressure_magnitude = jnp.abs(field_on_grid)
        
        # Get pressure at target point
        target_x_vox = N[0] // 2
        target_z_vox = int(FOCAL_DISTANCE_MM / dx_mm)
        target_pressure = pressure_magnitude[target_x_vox, target_z_vox]
        
        # Calculate relevant metrics based on objective type
        if OBJECTIVE_TYPE == "point":
            metric_value = target_pressure
            metric_label = f"Target Point Pressure: {metric_value:.3f}"
        elif OBJECTIVE_TYPE in ["area_contrast", "uniform", "sidelobe"]:
            # Calculate mean pressure in target area
            target_mask = create_circular_mask(
                center_x_mm=GRID_SIZE_MM[0]/2,
                center_z_mm=FOCAL_DISTANCE_MM,
                radius_mm=TARGET_RADIUS_MM
            )
            target_mask_float = target_mask.astype(jnp.float32)
            
            # Mean pressure using weighted sum
            target_sum = jnp.sum(pressure_magnitude * target_mask_float)
            target_count = jnp.sum(target_mask_float)
            mean_target_pressure = target_sum / (target_count + 1e-8)
            
            if OBJECTIVE_TYPE == "area_contrast":
                non_target_mask_float = 1.0 - target_mask_float
                non_target_sum = jnp.sum(pressure_magnitude * non_target_mask_float)
                non_target_count = jnp.sum(non_target_mask_float)
                non_target_pressure = non_target_sum / (non_target_count + 1e-8)
                contrast = mean_target_pressure / (non_target_pressure + 1e-8)
                metric_value = mean_target_pressure
                metric_label = f"Target Mean: {mean_target_pressure:.3f}, Contrast: {contrast:.1f}"
            elif OBJECTIVE_TYPE == "uniform":
                squared_diff = (pressure_magnitude - mean_target_pressure) ** 2
                variance_sum = jnp.sum(squared_diff * target_mask_float)
                variance = variance_sum / (target_count + 1e-8)
                metric_value = mean_target_pressure
                metric_label = f"Target Mean: {mean_target_pressure:.3f}, Var: {variance:.4f}"
            elif OBJECTIVE_TYPE == "sidelobe":
                sidelobe_region = create_circular_mask(
                    center_x_mm=GRID_SIZE_MM[0]/2,
                    center_z_mm=FOCAL_DISTANCE_MM,
                    radius_mm=SIDELOBE_RADIUS_MM
                )
                sidelobe_region_float = sidelobe_region.astype(jnp.float32)
                sidelobe_mask_float = sidelobe_region_float - target_mask_float
                
                sidelobe_sum = jnp.sum(pressure_magnitude * sidelobe_mask_float)
                sidelobe_count = jnp.sum(sidelobe_mask_float)
                sidelobe_pressure = sidelobe_sum / (sidelobe_count + 1e-8)
                metric_value = mean_target_pressure
                metric_label = f"Target: {mean_target_pressure:.3f}, Sidelobe: {sidelobe_pressure:.3f}"
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot the pressure field with consistent scaling
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = jnp.max(pressure_magnitude)
        
        im = plt.imshow(pressure_magnitude.T, 
                       extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                       aspect='auto',
                       cmap='hot',
                       origin='upper',
                       vmin=vmin,
                       vmax=VMAX)
        
        # Add colorbar
        plt.colorbar(im, label='Pressure Magnitude')
        
        # Add target point marker with pressure value
        target_x_mm = GRID_SIZE_MM[0] / 2
        target_z_mm = FOCAL_DISTANCE_MM
        plt.plot(target_x_mm, target_z_mm, 'b*', markersize=10, 
                label=metric_label)
        
        # Add lens region outline
        lens_x_start_mm = (GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2
        lens_x_end_mm = lens_x_start_mm + LENS_WIDTH_MM
        lens_z_start_mm = LENS_START_Z_MM
        lens_z_end_mm = lens_z_start_mm + LENS_THICKNESS_MM
        
        # Draw lens rectangle
        from matplotlib.patches import Rectangle
        lens_rect = Rectangle((lens_x_start_mm, lens_z_start_mm), 
                            LENS_WIDTH_MM, LENS_THICKNESS_MM,
                            linewidth=2, edgecolor='white', facecolor='none',
                            linestyle='--', label='Lens Region')
        plt.gca().add_patch(lens_rect)
        
        # Add natural focal point of bowl
        natural_focus_z = BOWL_ROC_MM
        plt.plot(target_x_mm, natural_focus_z, 'g+', markersize=10, 
                label=f'Natural Focus (z={natural_focus_z}mm)')
        
        # Add target area visualization for area-based objectives
        if OBJECTIVE_TYPE in ["area_contrast", "uniform", "sidelobe"]:
            from matplotlib.patches import Circle
            target_circle = Circle(
                (GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM),
                radius=TARGET_RADIUS_MM,
                linewidth=2,
                edgecolor='cyan',
                facecolor='none',
                linestyle='-',
                label=f'Target Area (r={TARGET_RADIUS_MM}mm)'
            )
            plt.gca().add_patch(target_circle)
            
            # Add sidelobe suppression region for sidelobe objective
            if OBJECTIVE_TYPE == "sidelobe":
                sidelobe_circle = Circle(
                    (GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM),
                    radius=SIDELOBE_RADIUS_MM,
                    linewidth=1.5,
                    edgecolor='yellow',
                    facecolor='none',
                    linestyle='--',
                    label=f'Sidelobe Region (r={SIDELOBE_RADIUS_MM}mm)'
                )
                plt.gca().add_patch(sidelobe_circle)
        
        plt.xlabel('X (mm)')
        plt.ylabel('Z (mm)')
        plt.title(f'Pressure Field - Iteration {iteration}')
        plt.legend()
        
        # Save the figure
        output_path = f'{OUTPUT_DIR}/pressure_field_iter_{iteration:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  -> Saved pressure field to {output_path}")
        return metric_value, vmax
    
    # Save initial pressure field and get max value for consistent scaling
    initial_params = get_params(opt_state)
    initial_target_pressure, initial_vmax = save_pressure_field(initial_params, 0)
    pressure_history.append(initial_target_pressure)
    
    # Use consistent color scaling across all iterations
    color_vmax = initial_vmax * 1.5  # Allow some headroom for improvements
    
    for i in range(n_iterations):
        iter_start = time.time()
        
        # Get current parameters for gradient check
        current_params = get_params(opt_state)
        
        # Compute loss and gradient
        loss, grad_val = objective_and_grad(current_params, bowl_source, domain)
        
        # Apply optimization step
        opt_state, loss = optimization_step(opt_state, bowl_source, domain, i)
        iter_time = time.time() - iter_start
        
        # Get updated parameters
        updated_params = get_params(opt_state)
        
        # Print debug info every iteration
        grad_norm = jnp.linalg.norm(grad_val)
        param_change = jnp.linalg.norm(updated_params - current_params)
        print(f"Iter {i+1}: Loss={loss:.4f}, |grad|={grad_norm:.4e}, |Δparams|={param_change:.4e}")
        
        # Save pressure field every 5 iterations
        if (i + 1) % 5 == 0:
            target_pressure, _ = save_pressure_field(updated_params, i + 1, vmin=0, vmax=VMAX)
            pressure_history.append(target_pressure)
    
    total_time = time.time() - start_time
    # Get the final optimized parameters
    optimized_params = get_params(opt_state)
    print("\nOptimization complete!")
    print(f"Final Loss: {loss:.4f}")
    print(f"Total optimization time: {total_time:.2f}s ({total_time/n_iterations:.2f}s per iteration)")
    
    # Save final pressure field
    final_target_pressure, _ = save_pressure_field(optimized_params, n_iterations, vmin=0, vmax=VMAX)
    pressure_history.append(final_target_pressure)
    
    # Print pressure history
    print("\nOptimization results:")
    if OBJECTIVE_TYPE == "point":
        print("Pressure at target point during optimization:")
    elif OBJECTIVE_TYPE == "area_contrast":
        print("Mean pressure in target area and contrast during optimization:")
    elif OBJECTIVE_TYPE == "uniform":
        print("Mean pressure and variance in target area during optimization:")
    elif OBJECTIVE_TYPE == "sidelobe":
        print("Mean pressure in target and sidelobe regions during optimization:")
    
    print(f"Initial: {pressure_history[0]:.4f}")
    print(f"Final: {pressure_history[-1]:.4f}")
    print(f"Change: {pressure_history[-1] - pressure_history[0]:.4f} ({(pressure_history[-1]/pressure_history[0] - 1)*100:.1f}%)")
    
    # You can now use `optimized_params` to generate the final sound speed map
    final_sos_map = create_optimizable_sos_map(optimized_params, domain)
    print("\nAn object `final_sos_map` containing the optimized sound speed distribution has been created.")
    # In a real application, you would save or visualize this map.
    # For example:
    plt.figure(figsize=(10, 6))
    sos_on_grid = final_sos_map.on_grid[..., 0]
    im = plt.imshow(sos_on_grid.T, cmap='viridis', 
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                   aspect='auto', origin='upper')
    plt.title("Optimized Sound Speed Map")
    plt.colorbar(im, label="Sound Speed (m/s)")
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    
    # Add lens region outline
    lens_x_start_mm = (GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2
    from matplotlib.patches import Rectangle
    lens_rect = Rectangle((lens_x_start_mm, LENS_START_Z_MM), 
                        LENS_WIDTH_MM, LENS_THICKNESS_MM,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', label='Lens Region')
    plt.gca().add_patch(lens_rect)
    
    plt.savefig(f'{OUTPUT_DIR}/optimized_sos_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved optimized sound speed map to /workspace/hologram/outs/optimized_sos_map.png")
    
    # Also save initial sound speed map for comparison
    initial_sos_map = create_optimizable_sos_map(initial_params, domain)
    plt.figure(figsize=(10, 6))
    sos_on_grid = initial_sos_map.on_grid[..., 0]
    im = plt.imshow(sos_on_grid.T, cmap='viridis', 
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[1], 0],
                   aspect='auto', origin='upper')
    plt.title("Initial Sound Speed Map")
    plt.colorbar(im, label="Sound Speed (m/s)")
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    
    lens_rect = Rectangle((lens_x_start_mm, LENS_START_Z_MM), 
                        LENS_WIDTH_MM, LENS_THICKNESS_MM,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', label='Lens Region')
    plt.gca().add_patch(lens_rect)
    
    plt.savefig(f'{OUTPUT_DIR}/initial_sos_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved initial sound speed map to /workspace/hologram/outs/initial_sos_map.png")
    
    # Save the optimized parameters for later analysis
    np.save(f'{OUTPUT_DIR}/optimized_params.npy', np.array(optimized_params))
    print("  -> Saved optimized parameters to /workspace/hologram/outs/optimized_params.npy")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test binarized acoustic lenses with positioning perturbations.

Usage: python try_lens.py <optimization_directory>
"""

import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import jax, jax.numpy as jnp
from jax import random, jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# Import export utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'codegen'))
from export_lenses_simple import (
    params_to_lens_designs, generate_openscad_script, 
    export_voxel_grid, export_point_cloud
)

# ---------------------------------------------------------------------------
#  JAX SETUP
# ---------------------------------------------------------------------------
jax.config.update("jax_default_matmul_precision", "bfloat16")
DTYPE = jnp.float32
CDTYPE = jnp.complex64

# ---------------------------------------------------------------------------
#  LOAD OPTIMIZATION RESULTS
# ---------------------------------------------------------------------------
def load_optimization_results(opt_dir):
    """Load parameters and metadata from optimization directory."""
    opt_dir = Path(opt_dir)
    params = np.load(opt_dir / "optimised_params.npy")
    with open(opt_dir / "run.json", 'r') as f:
        metadata = json.load(f)
    
    # Extract physical parameters
    phys = metadata['physical_params']
    
    return params, metadata, phys


# ---------------------------------------------------------------------------
#  BINARIZE LENS PARAMETERS
# ---------------------------------------------------------------------------
def binarize_params(params, sos_min, sos_max):
    """Binarize parameters to closest material (0 for min, 1 for max)."""
    # Apply sigmoid to get material fractions [0, 1]
    matfrac = jax.nn.sigmoid(params)
    
    # Binarize: round to nearest material
    binary = jnp.round(matfrac)
    
    # Convert back to parameter space (inverse sigmoid)
    # sigmoid^-1(x) = log(x / (1-x))
    # For binary values, use large negative/positive values
    binary_params = jnp.where(binary < 0.5, -10.0, 10.0)
    
    return binary_params


# ---------------------------------------------------------------------------
#  CREATE SIMULATION ENVIRONMENT
# ---------------------------------------------------------------------------
def setup_environment(phys):
    """Set up domain, bowl source, and other simulation parameters."""
    # Extract parameters
    FREQ_HZ = phys['freq_hz']
    BOWL_DIAM_MM = phys['bowl_diam_mm']
    BOWL_ROC_MM = phys['bowl_roc_mm']
    DX_MM = phys['dx_mm']
    GRID_SIZE_MM = phys['grid_size_mm']
    
    OMEGA = DTYPE(2 * np.pi * FREQ_HZ)
    DX_M = DTYPE(DX_MM / 1e3)
    N = tuple(int(d / DX_MM) for d in GRID_SIZE_MM)
    
    domain = Domain(N, tuple([DX_M] * 3))
    
    # Create bowl source
    x = (jnp.arange(N[0]) * DX_MM - GRID_SIZE_MM[0] / 2).astype(DTYPE)
    y = (jnp.arange(N[1]) * DX_MM - GRID_SIZE_MM[1] / 2).astype(DTYPE)
    z = (jnp.arange(N[2]) * DX_MM).astype(DTYPE)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    R = jnp.sqrt(X ** 2 + Y ** 2 + (Z - BOWL_ROC_MM) ** 2)
    
    mask = (
        (jnp.abs(R - BOWL_ROC_MM) < DX_MM) &
        (jnp.sqrt(X ** 2 + Y ** 2) <= (BOWL_DIAM_MM / 2)) &
        (Z < BOWL_ROC_MM)
    )
    
    phase = -OMEGA / phys['sound_speed_water'] * R
    field_complex = jnp.exp(1j * phase).astype(CDTYPE)
    zeros_complex = jnp.zeros_like(field_complex)
    
    src = jnp.where(mask, field_complex, zeros_complex)
    bowl_source = FourierSeries(src[..., None], domain)
    
    return domain, bowl_source, mask, OMEGA, N


# ---------------------------------------------------------------------------
#  PARAMS TO SOUND SPEED (WITH Z OFFSET)
# ---------------------------------------------------------------------------
def params_to_sos_offset(lens_params, phys, domain, z_offset_mm=0.0):
    """Convert parameters to sound speed field with Z offset."""
    # Extract all needed parameters
    NUM_LENSES = phys['num_lenses']
    NUM_XY_SEGMENTS = phys['num_xy_segments']
    LENS_WIDTH_MM = phys['lens_width_mm']
    DX_MM = phys['dx_mm']
    VOXEL_SIZE_MM = phys['voxel_size_mm']
    LENS_THICKNESS_MM = phys['lens_thickness_mm']
    BACKING_THICKNESS_MM = phys['backing_thickness_mm']
    GAP_BETWEEN_LENSES_MM = phys['gap_between_lenses_mm']
    LENS_START_Z_MM = phys['lens_start_z_mm']
    
    SOS_AGILUS = phys['sos_agilus']
    SOS_VEROCLR = phys['sos_veroclr']
    SOUND_SPEED_WATER = phys['sound_speed_water']
    
    N = domain.N
    
    # Apply sigmoid and reshape
    matfrac = jax.nn.sigmoid(
        lens_params.reshape(NUM_LENSES, NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
    ).astype(DTYPE)
    
    # Grid calculations
    lens_xy_start_vox = int(((phys['grid_size_mm'][0] - LENS_WIDTH_MM) / 2) / DX_MM)
    voxel_size_in_grid = int(VOXEL_SIZE_MM / DX_MM)
    lens_width_vox = NUM_XY_SEGMENTS * voxel_size_in_grid
    lens_thickness_vox = int(LENS_THICKNESS_MM / DX_MM)
    backing_vox = int(BACKING_THICKNESS_MM / DX_MM)
    gap_vox = int(GAP_BETWEEN_LENSES_MM / DX_MM)
    
    # Upsample
    matfrac = jnp.repeat(jnp.repeat(matfrac, voxel_size_in_grid, 1),
                         voxel_size_in_grid, 2)
    
    # Build lens stack
    opt_layer = jnp.repeat(matfrac[..., None], lens_thickness_vox, -1)
    backing = jnp.ones_like(opt_layer[..., :backing_vox])
    lens_block = jnp.concatenate([opt_layer, backing], -1)
    
    gap_block = jnp.zeros_like(opt_layer[..., :gap_vox])
    
    blocks = []
    for idx in range(NUM_LENSES):
        blocks.append(lens_block[idx])
        if idx < NUM_LENSES - 1:
            blocks.append(gap_block[idx])
    stack = jnp.concatenate(blocks, -1)
    
    # Calculate Z position with offset
    z_offset_vox = int(z_offset_mm / DX_MM)
    lens_start_z_vox = int(LENS_START_Z_MM / DX_MM) + z_offset_vox
    
    # Create sound speed field
    sos = jnp.ones(N, dtype=DTYPE) * SOUND_SPEED_WATER
    sos_lens = SOS_AGILUS + stack * (SOS_VEROCLR - SOS_AGILUS)
    
    sos = sos.at[
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_start_z_vox:lens_start_z_vox + stack.shape[-1],
    ].set(sos_lens)
    
    return FourierSeries(sos[..., None], domain)


# ---------------------------------------------------------------------------
#  FIELD SOLVER
# ---------------------------------------------------------------------------
def compute_field(lens_params, phys, domain, bowl_source, omega, z_offset_mm=0.0):
    """Compute pressure field with lens at given Z offset."""
    medium = Medium(
        domain=domain,
        sound_speed=params_to_sos_offset(lens_params, phys, domain, z_offset_mm),
        pml_size=10
    )
    return helmholtz_solver(medium, omega, bowl_source, tol=1e-3, checkpoint=True)


# ---------------------------------------------------------------------------
#  METRICS COMPUTATION
# ---------------------------------------------------------------------------
def compute_metrics(field, phys, domain):
    """Compute focal and sidelobe metrics."""
    N = domain.N
    DX_MM = phys['dx_mm']
    GRID_SIZE_MM = phys['grid_size_mm']
    FOCAL_DISTANCE_MM = phys['focal_distance_mm']
    TARGET_RADIUS_MM = phys['target_radius_mm']
    
    press = jnp.abs(field.on_grid[..., 0] if field.on_grid.ndim == 4 else field.on_grid)
    
    # Create masks
    x = jnp.arange(N[0]) * DX_MM
    y = jnp.arange(N[1]) * DX_MM
    z = jnp.arange(N[2]) * DX_MM
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Target mask
    target_mask = ((X - GRID_SIZE_MM[0] / 2) ** 2 +
                   (Y - GRID_SIZE_MM[1] / 2) ** 2 +
                   (Z - FOCAL_DISTANCE_MM) ** 2) <= TARGET_RADIUS_MM ** 2
    target_mask = target_mask.astype(DTYPE)
    
    # Sidelobe mask (if needed)
    sidelobe_mask = None
    if 'sidelobe_radius_mm' in phys and phys['sidelobe_radius_mm'] is not None:
        SIDELOBE_RADIUS_MM = phys['sidelobe_radius_mm']
        sidelobe_region = ((X - GRID_SIZE_MM[0] / 2) ** 2 +
                          (Y - GRID_SIZE_MM[1] / 2) ** 2 +
                          (Z - FOCAL_DISTANCE_MM) ** 2) <= SIDELOBE_RADIUS_MM ** 2
        sidelobe_mask = (sidelobe_region & ~target_mask).astype(DTYPE)
    
    # Compute metrics
    target_pressure = float((press * target_mask).sum() / (target_mask.sum() + 1e-8))
    
    sidelobe_pressure = None
    if sidelobe_mask is not None:
        sidelobe_pressure = float((press * sidelobe_mask).sum() / (sidelobe_mask.sum() + 1e-8))
    
    return target_pressure, sidelobe_pressure


# ---------------------------------------------------------------------------
#  VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def save_pressure_xz(field, phys, domain, tag, output_dir):
    """Save XZ slice of pressure field."""
    N = domain.N
    DX_MM = phys['dx_mm']
    GRID_SIZE_MM = phys['grid_size_mm']
    FOCAL_DISTANCE_MM = phys['focal_distance_mm']
    
    y_mid = N[1] // 2
    press = np.abs(np.asarray(field.on_grid[..., 0] if field.on_grid.ndim == 4 else field.on_grid))
    
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(press[:, y_mid, :].T,
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[2], 0],
                   cmap='hot', aspect='auto', origin='upper', vmin=0, vmax=1.0)
    ax.set_title(f"|p| - {tag} (XZ slice)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    fig.colorbar(im, ax=ax, label='|p|')
    
    # Mark focal point
    ax.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'gx', ms=10, mew=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"pressure_{tag}.png", dpi=200)
    plt.close()


def save_lens_png(lens_params, phys, fname):
    """Save lens designs as PNG."""
    NUM_LENSES = phys['num_lenses']
    NUM_XY_SEGMENTS = phys['num_xy_segments']
    LENS_WIDTH_MM = phys['lens_width_mm']
    TOTAL_VOXELS_PER_LENS = phys['total_voxels_per_lens']
    
    fig, ax = plt.subplots(1, NUM_LENSES, figsize=(5 * NUM_LENSES, 4))
    if NUM_LENSES == 1:
        ax = [ax]
    
    for k in range(NUM_LENSES):
        seg = lens_params[k * TOTAL_VOXELS_PER_LENS:(k + 1) * TOTAL_VOXELS_PER_LENS]
        seg = jax.nn.sigmoid(seg).reshape(NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
        im = ax[k].imshow(seg, vmin=0, vmax=1, cmap='RdBu_r',
                         extent=[0, LENS_WIDTH_MM, 0, LENS_WIDTH_MM])
        ax[k].set_title(f"Lens {k+1}")
        ax[k].set_xlabel("mm")
        ax[k].set_ylabel("mm")
        cbar = plt.colorbar(im, ax=ax[k])
        cbar.set_label('Material fraction\n(0=Agilus30, 1=VeroClear)')
    
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
#  MAIN FUNCTION
# ---------------------------------------------------------------------------
def main(opt_dir):
    """Main testing function."""
    opt_dir = Path(opt_dir)
    print(f"Loading optimization results from: {opt_dir}")
    
    # Load results
    params, metadata, phys = load_optimization_results(opt_dir)
    
    # Create output directory
    output_dir = Path(f"test_binarized_{opt_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set up environment
    domain, bowl_source, bowl_mask, omega, N = setup_environment(phys)
    print(f"Domain size: {N}")
    
    # Get material sound speeds
    SOS_AGILUS = phys['sos_agilus']
    SOS_VEROCLR = phys['sos_veroclr']
    print(f"Materials: Agilus30={SOS_AGILUS} m/s, VeroClear={SOS_VEROCLR} m/s")
    
    # Binarize parameters
    binary_params = binarize_params(params, SOS_AGILUS, SOS_VEROCLR)
    
    # Save lens visualization
    print("\nSaving lens designs...")
    save_lens_png(params, phys, output_dir / "original_lenses.png")
    save_lens_png(binary_params, phys, output_dir / "binarized_lenses.png")
    
    # Compute baseline (no lens)
    print("\nComputing baseline (no lens)...")
    print("  Creating parameters...")
    baseline_params = jnp.full_like(params, -10.0)  # All water
    print("  Running simulation...")
    baseline_field = compute_field(baseline_params, phys, domain, bowl_source, omega)
    print("  Computing metrics...")
    baseline_focal, baseline_sidelobe = compute_metrics(baseline_field, phys, domain)
    print("  Saving visualization...")
    save_pressure_xz(baseline_field, phys, domain, "baseline", output_dir)
    print(f"  Baseline focal pressure: {baseline_focal:.6f}")
    
    # Compute with original optimized lens
    print("\nComputing with original lens...")
    print("  Running simulation...")
    orig_field = compute_field(params, phys, domain, bowl_source, omega)
    print("  Computing metrics...")
    orig_focal, orig_sidelobe = compute_metrics(orig_field, phys, domain)
    print("  Saving visualization...")
    save_pressure_xz(orig_field, phys, domain, "original", output_dir)
    print(f"  Original focal pressure: {orig_focal:.6f}")
    
    # Compute with binarized lens
    print("\nComputing with binarized lens...")
    print("  Running simulation...")
    binary_field = compute_field(binary_params, phys, domain, bowl_source, omega)
    print("  Computing metrics...")
    binary_focal, binary_sidelobe = compute_metrics(binary_field, phys, domain)
    print("  Saving visualization...")
    save_pressure_xz(binary_field, phys, domain, "binarized", output_dir)
    print(f"  Binarized focal pressure: {binary_focal:.6f}")
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Focal pressure (mean |p| in target region):")
    print(f"  Baseline:  {baseline_focal:.6f}")
    print(f"  Original:  {orig_focal:.6f} (+{(orig_focal/baseline_focal-1)*100:.1f}%)")
    print(f"  Binarized: {binary_focal:.6f} (+{(binary_focal/baseline_focal-1)*100:.1f}%)")
    print(f"  Binary vs Original: {(binary_focal/orig_focal)*100:.1f}% retained")
    
    if baseline_sidelobe is not None:
        print(f"\nSidelobe pressure:")
        print(f"  Baseline:  {baseline_sidelobe:.6f}")
        print(f"  Original:  {orig_sidelobe:.6f} ({(orig_sidelobe/baseline_sidelobe)*100:.1f}%)")
        print(f"  Binarized: {binary_sidelobe:.6f} ({(binary_sidelobe/baseline_sidelobe)*100:.1f}%)")
    
    # Test positioning robustness
    print("\n" + "="*60)
    print("POSITIONING ROBUSTNESS TEST")
    print("="*60)
    print("Testing with random Z offsets within ±1mm...")
    
    n_tests = 5
    offsets_mm = []
    focal_pressures = []
    sidelobe_pressures = []
    
    key = random.PRNGKey(42)
    for i in range(n_tests):
        key, subkey = random.split(key)
        offset_mm = float(random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        offsets_mm.append(offset_mm)
        
        field = compute_field(binary_params, phys, domain, bowl_source, omega, offset_mm)
        focal, sidelobe = compute_metrics(field, phys, domain)
        focal_pressures.append(focal)
        if sidelobe is not None:
            sidelobe_pressures.append(sidelobe)
        
        print(f"  Offset {offset_mm:+.3f}mm: focal={focal:.6f} ({(focal/binary_focal)*100:.1f}%)")
        
        # Save visualization for first offset
        if i == 0:
            save_pressure_xz(field, phys, domain, f"offset_{offset_mm:+.3f}mm", output_dir)
    
    # Statistics
    focal_mean = np.mean(focal_pressures)
    focal_std = np.std(focal_pressures)
    print(f"\nFocal pressure statistics:")
    print(f"  Mean:  {focal_mean:.6f} ({(focal_mean/binary_focal)*100:.1f}% of nominal)")
    print(f"  Std:   {focal_std:.6f} ({(focal_std/focal_mean)*100:.1f}% variation)")
    print(f"  Range: [{min(focal_pressures):.6f}, {max(focal_pressures):.6f}]")
    
    # Export 3D models
    print("\n" + "="*60)
    print("EXPORTING 3D MODELS")
    print("="*60)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set global variables needed by export functions
    import export_lenses_simple
    export_lenses_simple.NUM_LENSES = phys['num_lenses']
    export_lenses_simple.LENS_WIDTH_MM = phys['lens_width_mm']
    export_lenses_simple.LENS_THICKNESS_MM = phys['lens_thickness_mm']
    export_lenses_simple.BACKING_THICKNESS_MM = phys['backing_thickness_mm']
    export_lenses_simple.GAP_BETWEEN_LENSES_MM = phys['gap_between_lenses_mm']
    export_lenses_simple.VOXEL_SIZE_MM = phys['voxel_size_mm']
    export_lenses_simple.NUM_XY_SEGMENTS = phys['num_xy_segments']
    export_lenses_simple.TOTAL_VOXELS_PER_LENS = phys['total_voxels_per_lens']
    
    # Convert to binary designs
    binary_designs = params_to_lens_designs(binary_params)
    
    # Export voxel grids
    export_voxel_grid(binary_designs, models_dir)
    print(f"  Exported voxel grids to {models_dir}")
    
    # Generate OpenSCAD script
    openscad_file = models_dir / "lenses.scad"
    generate_openscad_script(binary_designs, openscad_file)
    print(f"  Generated OpenSCAD script: {openscad_file}")
    
    # Export point clouds
    z_offset = 0
    for i in range(phys['num_lenses']):
        filename = models_dir / f"lens_{i+1}_points.txt"
        n_points = export_point_cloud(binary_designs[i], filename, z_offset)
        z_offset += phys['lens_thickness_mm'] + phys['backing_thickness_mm'] + phys['gap_between_lenses_mm']
    print(f"  Exported point clouds to {models_dir}")
    
    # Export multi-material files for PolyJet
    print("\nExporting PolyJet multi-material files...")
    export_lenses_simple.export_multi_material_stl(binary_designs, models_dir)
    
    # Save summary
    summary = {
        "optimization_dir": str(opt_dir),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "performance": {
            "baseline_focal": baseline_focal,
            "original_focal": orig_focal,
            "binarized_focal": binary_focal,
            "retention_percent": (binary_focal/orig_focal)*100,
            "improvement_over_baseline": (binary_focal/baseline_focal-1)*100
        },
        "robustness": {
            "offsets_mm": offsets_mm,
            "focal_pressures": focal_pressures,
            "focal_mean": focal_mean,
            "focal_std": focal_std,
            "focal_cv_percent": (focal_std/focal_mean)*100
        },
        "physical_params": phys
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nTo generate STL files:")
    print(f"  cd {models_dir}")
    print("  openscad -o assembly.stl lenses.scad")
    print("\nFor PolyJet multi-material printing:")
    print(f"  cd {models_dir}/multi_material_stl")
    print("  bash generate_stls.sh")
    print("  Import all _RGD_*.stl files into GrabCAD Print as an assembly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test binarized acoustic lenses with positioning perturbations"
    )
    parser.add_argument("opt_dir", help="Path to optimization results directory")
    args = parser.parse_args()
    
    main(args.opt_dir) 
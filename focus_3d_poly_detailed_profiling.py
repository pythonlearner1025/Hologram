#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced profiling script with detailed timing breakdowns for optimization operations.
"""

import os, time, json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from jax import custom_vjp
from functools import wraps
from collections import defaultdict

import jax, jax.numpy as jnp
from jax import random, value_and_grad, jit
from jax.experimental import mesh_utils
import optax

from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# Import adjoint utilities
from adjoint_utils import solve_adjoint_precond

# ---------------------------------------------------------------------------
#  ENHANCED PROFILING UTILITIES
# ---------------------------------------------------------------------------
class DetailedTimer:
    """Enhanced timer with nested timing support."""
    def __init__(self, name, timings_dict, parent=None):
        self.name = name
        self.timings_dict = timings_dict
        self.parent = parent
        self.start = None
        self.children_time = 0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        # Store raw elapsed time
        if self.name not in self.timings_dict:
            self.timings_dict[self.name] = []
        self.timings_dict[self.name].append(elapsed)
        
        # If this timer has a parent, add its time to parent's children time
        if self.parent:
            self.parent.children_time += elapsed

# Global timing storage
timings = defaultdict(list)
timing_stack = []  # Stack to track nested timers

def timer(name):
    """Create a timer with parent tracking."""
    parent = timing_stack[-1] if timing_stack else None
    t = DetailedTimer(name, timings, parent)
    timing_stack.append(t)
    return t

def exit_timer():
    """Exit the current timer."""
    if timing_stack:
        timing_stack.pop()

# ---------------------------------------------------------------------------
#  ARGUMENT PARSER
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Profile acoustic lens optimization with detailed breakdown')
parser.add_argument('--freq-hz', type=float, default=1.5e6)
parser.add_argument('--bowl-diam-mm', type=float, default=25.0)
parser.add_argument('--bowl-roc-mm', type=float, default=30.0)
parser.add_argument('--focal-distance-mm', type=float, default=None)
parser.add_argument('--num-lenses', type=int, default=5)
parser.add_argument('--profile-iters', type=int, default=2,
                    help='Number of iterations to profile (default: 2)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
#  JAX & mixed‑precision set‑up
# ---------------------------------------------------------------------------
jax.config.update("jax_default_matmul_precision", "bfloat16")
DTYPE  = jnp.float32
CDTYPE = jnp.complex64
key    = random.PRNGKey(42)

# ---------------------------------------------------------------------------
#  CONSTANTS (same as main script)
# ---------------------------------------------------------------------------
FREQ_HZ            = args.freq_hz
FREQ_MHZ           = FREQ_HZ / 1e6
BOWL_DIAM_MM = args.bowl_diam_mm
BOWL_ROC_MM = args.bowl_roc_mm
NUM_LENSES = args.num_lenses
transducer_spec = f"OD{int(BOWL_DIAM_MM)}_F{FREQ_MHZ:.1f}MHz_R{int(BOWL_ROC_MM)}_L{NUM_LENSES}"
OUTDIR = f"/workspace/hologram/test/detailed_profiling_{transducer_spec}_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTDIR, exist_ok=True)
OMEGA              = DTYPE(2 * np.pi * FREQ_HZ)

# Material properties
SOUND_SPEED_WATER  = DTYPE(1500.0)
SOS_AGILUS         = DTYPE(2035.0)
SOS_VEROCLR        = DTYPE(2473.0)

FOCAL_DISTANCE_MM = args.focal_distance_mm if args.focal_distance_mm is not None else BOWL_ROC_MM * 1.08
TARGET_RADIUS_MM = 0.5
SIDELOBE_RADIUS_MM = 4.0

LENS_WIDTH_MM         = BOWL_DIAM_MM
LENS_THICKNESS_MM     = 0.5
BACKING_THICKNESS_MM  = 0.5
GAP_BETWEEN_LENSES_MM = 0.5

PPW = 6
VOXEL_SIZE_MM = 0.5
DX_MM         =  min(0.125, (1500/FREQ_HZ/PPW)*1e3)
DX_M          = DTYPE(DX_MM / 1e3)

GREM_TOL = 1e-3

GRID_SIZE_MM = (BOWL_DIAM_MM, BOWL_DIAM_MM, BOWL_ROC_MM*1.2)
N = tuple(int(d / DX_MM) for d in GRID_SIZE_MM)

domain = Domain(N, tuple([DX_M] * 3))
print(f"Domain voxels: {N} = {np.prod(N):,} total voxels")

# Static computations
NUM_XY_SEGMENTS       = int(LENS_WIDTH_MM / VOXEL_SIZE_MM)
TOTAL_VOXELS_PER_LENS = NUM_XY_SEGMENTS ** 2

lens_xy_start_vox   = int(((GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2) / DX_MM)
voxel_size_in_grid  = int(VOXEL_SIZE_MM / DX_MM)
lens_width_vox      = NUM_XY_SEGMENTS * voxel_size_in_grid
lens_thickness_vox  = int(LENS_THICKNESS_MM / DX_MM)
backing_vox         = int(BACKING_THICKNESS_MM / DX_MM)
gap_vox             = int(GAP_BETWEEN_LENSES_MM / DX_MM)

USE_SIDELOBE_OBJECTIVE = True
SIDELOBE_PENALTY = 0.5

# ---------------------------------------------------------------------------
#  SOURCE (bowl transducer)
# ---------------------------------------------------------------------------
def create_bowl_source_3d() -> FourierSeries:
    """Return a complex pressure distribution on the transducer surface."""
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

    phase          = -OMEGA / SOUND_SPEED_WATER * R
    field_complex  = jnp.exp(1j * phase).astype(CDTYPE)
    zeros_complex  = jnp.zeros_like(field_complex)

    src = jnp.where(mask, field_complex, zeros_complex)
    return FourierSeries(src[..., None], domain), mask

bowl_source, bowl_mask = create_bowl_source_3d()

z_indices = jnp.arange(N[2])
max_z_vox = int(jnp.max(jnp.where(bowl_mask.any(axis=(0, 1)), z_indices, 0)))
MAX_BOWL_Z_MM = max_z_vox * DX_MM

LENS_START_Z_MM = MAX_BOWL_Z_MM + 1.0
lens_start_z_vox = int(LENS_START_Z_MM / DX_MM)

# ---------------------------------------------------------------------------
#  PROFILED PARAMS → SOUND‑SPEED FIELD
# ---------------------------------------------------------------------------
def _params_to_sos_inner(lens_params: jnp.ndarray) -> FourierSeries:
    """Inner function for params_to_sos without JIT decoration."""
    # Reshape and apply sigmoid
    matfrac = jax.nn.sigmoid(
        lens_params.reshape(NUM_LENSES, NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
    ).astype(DTYPE)
    
    # Upsample
    matfrac = jnp.repeat(jnp.repeat(matfrac, voxel_size_in_grid, 1),
                         voxel_size_in_grid, 2)
    
    # Build lens blocks
    opt_layer  = jnp.repeat(matfrac[..., None], lens_thickness_vox, -1)
    backing    = jnp.ones_like(opt_layer[..., :backing_vox])
    lens_block = jnp.concatenate([opt_layer, backing], -1)
    gap_block  = jnp.zeros_like(opt_layer[..., :gap_vox])
    
    blocks = []
    for idx in range(NUM_LENSES):
        blocks.append(lens_block[idx])
        if idx < NUM_LENSES - 1:
            blocks.append(gap_block[idx])
    stack = jnp.concatenate(blocks, -1)
    
    # Create sound speed field
    sos = jnp.ones(N, dtype=DTYPE) * SOUND_SPEED_WATER
    sos_lens = SOS_AGILUS + stack * (SOS_VEROCLR - SOS_AGILUS)
    
    sos = sos.at[
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_start_z_vox:lens_start_z_vox + stack.shape[-1],
    ].set(sos_lens)
    
    return FourierSeries(sos[..., None], domain)

# JIT compile the function
params_to_sos = jit(_params_to_sos_inner)

# ---------------------------------------------------------------------------
#  DETAILED COMPUTE FIELD
# ---------------------------------------------------------------------------
def _compute_field_inner(lens_params: jnp.ndarray):
    """Inner compute field function without JIT."""
    medium = Medium(domain=domain,
                    sound_speed=params_to_sos(lens_params),
                    pml_size=10)
    return helmholtz_solver(medium, OMEGA, bowl_source,
                            tol=GREM_TOL, checkpoint=True)

compute_field = jit(_compute_field_inner)

# ---------------------------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def create_spherical_mask_vectorized(center_x_mm, center_y_mm, center_z_mm, radius_mm):
    """Create a spherical mask centered at (center_x, center_y, center_z) with given radius."""
    x = jnp.arange(N[0]) * DX_MM
    y = jnp.arange(N[1]) * DX_MM
    z = jnp.arange(N[2]) * DX_MM
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    distance_squared = ((X - center_x_mm) ** 2 + 
                       (Y - center_y_mm) ** 2 + 
                       (Z - center_z_mm) ** 2)
    
    return distance_squared <= radius_mm ** 2

# ---------------------------------------------------------------------------
#  DETAILED SIDELOBE OBJECTIVE
# ---------------------------------------------------------------------------
@custom_vjp
def objective_sidelobe(lens_params: jnp.ndarray):
    """Maximize pressure in target area while suppressing sidelobes."""
    field = compute_field(lens_params).on_grid
    
    if field.ndim == 4:
        pressure_magnitude = jnp.abs(field[..., 0])
    else:
        pressure_magnitude = jnp.abs(field)
    
    target_mask = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_region = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, SIDELOBE_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_mask = sidelobe_region - target_mask
    
    target_pressure = jnp.sum(pressure_magnitude * target_mask) / (jnp.sum(target_mask) + 1e-8)
    sidelobe_pressure = jnp.sum(pressure_magnitude * sidelobe_mask) / (jnp.sum(sidelobe_mask) + 1e-8)
    
    return -(target_pressure - SIDELOBE_PENALTY * sidelobe_pressure), field

def objective_sidelobe_fwd(lens_params):
    """Forward pass with detailed profiling."""
    field = compute_field(lens_params).on_grid
    
    if field.ndim == 4:
        pressure_magnitude = jnp.abs(field[..., 0])
        field_slice = field[..., 0]
    else:
        pressure_magnitude = jnp.abs(field)
        field_slice = field
    
    target_mask = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_region = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, SIDELOBE_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_mask = sidelobe_region - target_mask
    
    target_pressure = jnp.sum(pressure_magnitude * target_mask) / (jnp.sum(target_mask) + 1e-8)
    sidelobe_pressure = jnp.sum(pressure_magnitude * sidelobe_mask) / (jnp.sum(sidelobe_mask) + 1e-8)
    loss = -(target_pressure - SIDELOBE_PENALTY * sidelobe_pressure)
    
    return loss, (field_slice, target_mask, sidelobe_mask, lens_params)

def objective_sidelobe_bwd(res, g):
    """Backward pass with detailed profiling."""
    field, target_mask, sidelobe_mask, lens_params = res
    g_loss = g
    
    # Gradient w.r.t. field
    eps = 1e-8
    field_mag = jnp.abs(field) + eps
    
    target_sum = jnp.sum(target_mask) + eps
    sidelobe_sum = jnp.sum(sidelobe_mask) + eps
    
    dl_dmag = -g_loss * (target_mask / target_sum - SIDELOBE_PENALTY * sidelobe_mask / sidelobe_sum)
    dl_du = 0.5 * dl_dmag * jnp.conj(field) / field_mag
    
    # Setup for adjoint
    medium = Medium(domain=domain,
                    sound_speed=params_to_sos(lens_params),
                    pml_size=10)
    
    # Solve adjoint
    if dl_du.ndim == 3:
        dl_du = dl_du[..., None]
    
    dl_du_conj = jnp.conj(dl_du[..., 0] if dl_du.ndim == 4 else dl_du)
    lambda_field = solve_adjoint_precond(
        dl_du_conj=dl_du_conj,
        domain=domain,
        fwd_medium=medium,
        omega=OMEGA,
        tol=GREM_TOL
    )
    
    # Gradient w.r.t. params
    def sos_map(params):
        return params_to_sos(params).on_grid[..., 0]
    
    _, vjp_sos = jax.vjp(sos_map, lens_params)
    
    c = params_to_sos(lens_params).on_grid[..., 0]
    dk2_dc = -2 * OMEGA**2 / (c**3)
    
    lambda_val = lambda_field.on_grid[..., 0] if lambda_field.on_grid.ndim == 4 else lambda_field.on_grid
    integrand = jnp.real(jnp.conj(lambda_val) * field * dk2_dc)
    
    grad_lens_params = vjp_sos(integrand)[0]
    
    return (grad_lens_params,)

objective_sidelobe.defvjp(objective_sidelobe_fwd, objective_sidelobe_bwd)

# ---------------------------------------------------------------------------
#  MANUALLY PROFILED FUNCTIONS
# ---------------------------------------------------------------------------
def profiled_objective_and_grad(params):
    """Manually profiled version of objective_and_grad."""
    # Forward pass
    with timer("forward_pass.total"):
        with timer("forward_pass.params_to_sos"):
            sos_field = params_to_sos(params)
            sos_field.on_grid.block_until_ready()
        
        with timer("forward_pass.medium_setup"):
            medium = Medium(domain=domain, sound_speed=sos_field, pml_size=10)
        
        with timer("forward_pass.helmholtz_solver"):
            field = helmholtz_solver(medium, OMEGA, bowl_source, tol=GREM_TOL, checkpoint=True)
            field.on_grid.block_until_ready()
        
        with timer("forward_pass.objective_calc"):
            pressure_magnitude = jnp.abs(field.on_grid[..., 0])
            
            target_mask = create_spherical_mask_vectorized(
                GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
            ).astype(DTYPE)
            
            sidelobe_region = create_spherical_mask_vectorized(
                GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, SIDELOBE_RADIUS_MM
            ).astype(DTYPE)
            
            sidelobe_mask = sidelobe_region - target_mask
            
            target_pressure = jnp.sum(pressure_magnitude * target_mask) / (jnp.sum(target_mask) + 1e-8)
            sidelobe_pressure = jnp.sum(pressure_magnitude * sidelobe_mask) / (jnp.sum(sidelobe_mask) + 1e-8)
            loss = -(target_pressure - SIDELOBE_PENALTY * sidelobe_pressure)
            loss.block_until_ready()
    
    # Backward pass
    with timer("backward_pass.total"):
        # Use JAX's automatic differentiation
        _, grads = value_and_grad(objective_sidelobe)(params)
        grads.block_until_ready()
    
    return loss, grads

# Use the JIT-compiled version for actual optimization
objective_sidelobe_and_grad = jit(value_and_grad(objective_sidelobe))

def profiled_optimization_step(params, opt_state):
    """Manually profiled optimization step."""
    with timer("optimization_step.total"):
        # For the first iteration, use manual profiling
        if len(timings) < 10:  # Only profile first iteration in detail
            loss, grads = profiled_objective_and_grad(params)
        else:
            # Use normal JIT version for subsequent iterations
            with timer("optimization_step.objective_and_grad"):
                loss, grads = objective_sidelobe_and_grad(params)
                loss.block_until_ready()
        
        with timer("optimization_step.optimizer_update"):
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params.block_until_ready()
    
    return params, opt_state, loss

# ---------------------------------------------------------------------------
#  INITIALIZATION
# ---------------------------------------------------------------------------
learning_rate = 0.3
optimiser     = optax.adamw(learning_rate)
total_params  = NUM_LENSES * TOTAL_VOXELS_PER_LENS
params        = 0.2 * (random.uniform(key, (total_params,), dtype=DTYPE) - 0.5)
opt_state     = optimiser.init(params)

# ---------------------------------------------------------------------------
#  WARM-UP
# ---------------------------------------------------------------------------
print("\n=== WARM-UP PHASE (JIT compilation) ===")
warmup_start = time.perf_counter()

# Run one iteration to compile all functions
loss_warmup, grads_warmup = objective_sidelobe_and_grad(params)

# Also warm up the compute_field function
_ = compute_field(params)

warmup_time = time.perf_counter() - warmup_start
print(f"Warm-up completed in {warmup_time:.2f}s")

# Clear warmup timings
timings.clear()

# ---------------------------------------------------------------------------
#  DETAILED PROFILING LOOP
# ---------------------------------------------------------------------------
print(f"\n=== DETAILED PROFILING FOR {args.profile_iters} ITERATIONS ===")
N_PROFILE_ITERS = args.profile_iters
losses = []

profile_start = time.perf_counter()

for i in range(N_PROFILE_ITERS):
    iter_start = time.perf_counter()
    
    with timer(f"iteration_{i}.total"):
        params, opt_state, loss = profiled_optimization_step(params, opt_state)
        losses.append(float(loss))
    
    iter_time = time.perf_counter() - iter_start
    print(f"\nIteration {i}: {iter_time:.3f}s | loss: {loss:.6f}")

total_profile_time = time.perf_counter() - profile_start

# ---------------------------------------------------------------------------
#  DETAILED TIMING ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("DETAILED TIMING BREAKDOWN")
print("="*80)

print(f"\nTotal profiling time: {total_profile_time:.2f}s")
print(f"Average time per iteration: {total_profile_time/N_PROFILE_ITERS:.2f}s")

# Calculate statistics
timing_summary = {}
for key, values in timings.items():
    timing_summary[key] = {
        'mean': np.mean(values),
        'std': np.std(values) if len(values) > 1 else 0,
        'min': np.min(values),
        'max': np.max(values),
        'total': np.sum(values),
        'count': len(values)
    }

# Hierarchical view
print("\n" + "-"*80)
print("HIERARCHICAL TIMING VIEW (First Iteration Detail)")
print("-"*80)

# Extract first iteration detailed timings
first_iter_timings = {k: v for k, v in timing_summary.items() 
                      if 'forward_pass' in k or 'backward_pass' in k}

if first_iter_timings:
    # Forward pass breakdown
    print("\nForward Pass:")
    fwd_total = timing_summary.get('forward_pass.total', {}).get('mean', 0)
    print(f"  Total: {fwd_total:.3f}s")
    
    fwd_components = [
        ('params_to_sos', 'forward_pass.params_to_sos'),
        ('medium_setup', 'forward_pass.medium_setup'),
        ('helmholtz_solver', 'forward_pass.helmholtz_solver'),
        ('objective_calc', 'forward_pass.objective_calc')
    ]
    
    for name, key in fwd_components:
        if key in timing_summary:
            t = timing_summary[key]['mean']
            pct = (t / fwd_total * 100) if fwd_total > 0 else 0
            print(f"    {name:<20}: {t:>8.3f}s ({pct:>5.1f}%)")
    
    # Backward pass breakdown
    print("\nBackward Pass:")
    bwd_total = timing_summary.get('backward_pass.total', {}).get('mean', 0)
    print(f"  Total: {bwd_total:.3f}s")

# Summary statistics
print("\n" + "-"*80)
print("OPERATION SUMMARY (All Iterations)")
print("-"*80)

# Group operations by category
categories = {
    'Optimization Step': ['optimization_step.total', 'optimization_step.objective_and_grad', 
                         'optimization_step.optimizer_update'],
    'Forward Pass': ['forward_pass.total', 'forward_pass.params_to_sos', 
                     'forward_pass.helmholtz_solver', 'forward_pass.objective_calc'],
    'Backward Pass': ['backward_pass.total'],
    'Iterations': [k for k in timing_summary.keys() if k.startswith('iteration_')]
}

for category, keys in categories.items():
    relevant_keys = [k for k in keys if k in timing_summary]
    if relevant_keys:
        print(f"\n{category}:")
        for key in relevant_keys:
            stats = timing_summary[key]
            print(f"  {key:<40}: {stats['mean']:>8.3f}s "
                  f"(total: {stats['total']:>8.3f}s, count: {stats['count']})")

# Performance insights
print("\n" + "="*80)
print("PERFORMANCE INSIGHTS")
print("="*80)

if 'forward_pass.helmholtz_solver' in timing_summary:
    helmholtz_time = timing_summary['forward_pass.helmholtz_solver']['mean']
    iter_time = total_profile_time / N_PROFILE_ITERS
    print(f"\n1. Helmholtz solver: {helmholtz_time:.3f}s ({helmholtz_time/iter_time*100:.1f}% of iteration)")
    print(f"   - Grid size: {N} = {np.prod(N):,} voxels")
    print(f"   - Voxels/second: {np.prod(N)/helmholtz_time:,.0f}")

if 'optimization_step.optimizer_update' in timing_summary:
    opt_time = timing_summary['optimization_step.optimizer_update']['mean']
    print(f"\n2. Optimizer update: {opt_time:.3f}s")
    print(f"   - Parameters: {total_params:,}")
    print(f"   - Updates/second: {total_params/opt_time:,.0f}")

# Save results
profile_data = {
    'config': {
        'num_lenses': NUM_LENSES,
        'grid_size': N,
        'total_voxels': int(np.prod(N)),
        'total_params': total_params,
        'freq_hz': FREQ_HZ,
        'profile_iterations': N_PROFILE_ITERS,
    },
    'timing_summary': timing_summary,
    'raw_timings': dict(timings),
    'total_time': total_profile_time,
    'warmup_time': warmup_time,
    'avg_iter_time': total_profile_time / N_PROFILE_ITERS
}

with open(f"{OUTDIR}/detailed_profiling_results.json", "w") as f:
    json.dump(profile_data, f, indent=2)

print(f"\nDetailed profiling results saved to: {OUTDIR}/detailed_profiling_results.json")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Main operation breakdown
if 'forward_pass.total' in timing_summary and 'backward_pass.total' in timing_summary:
    fwd_time = timing_summary['forward_pass.total']['mean']
    bwd_time = timing_summary['backward_pass.total']['mean']
    opt_time = timing_summary.get('optimization_step.optimizer_update', {}).get('mean', 0)
    
    labels = ['Forward Pass', 'Backward Pass', 'Optimizer Update']
    times = [fwd_time, bwd_time, opt_time]
    ax1.pie(times, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Time Distribution per Iteration')

# Forward pass breakdown
fwd_breakdown = []
fwd_labels = []
for name, key in [('Params→SOS', 'forward_pass.params_to_sos'),
                  ('Helmholtz', 'forward_pass.helmholtz_solver'),
                  ('Objective', 'forward_pass.objective_calc'),
                  ('Other', 'forward_pass.medium_setup')]:
    if key in timing_summary:
        fwd_breakdown.append(timing_summary[key]['mean'])
        fwd_labels.append(name)

if fwd_breakdown:
    ax2.pie(fwd_breakdown, labels=fwd_labels, autopct='%1.1f%%')
    ax2.set_title('Forward Pass Breakdown')

plt.tight_layout()
plt.savefig(f"{OUTDIR}/detailed_timing_breakdown.png", dpi=200)
plt.close()

print(f"Timing visualization saved to: {OUTDIR}/detailed_timing_breakdown.png") 
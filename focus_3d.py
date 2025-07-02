#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast 3‑D stacked‑lens optimiser for Formlabs‑printable acoustic devices.

Key changes vs. the reference implementation
--------------------------------------------
1.  Vectorised `params_to_sos()` removes 900 inner `.at[…].set()` calls.
2.  Pure‑JAX optimiser (Optax AdamW).
3.  Mixed precision: bfloat16 matmuls, float32 state.
4.  Cached Helmholtz trace (no recompiles across iterations).
"""

import os, time, json
import numpy as np
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
from jax import random, value_and_grad, jit
from jax.experimental import mesh_utils
import optax

from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# ---------------------------------------------------------------------------
#  JAX & mixed‑precision set‑up
# ---------------------------------------------------------------------------
jax.config.update("jax_default_matmul_precision", "bfloat16")   # Hopper/H200 is fast here
DTYPE  = jnp.float32          # main compute dtype
CDTYPE = jnp.complex64        # complex for frequency‑domain fields
key    = random.PRNGKey(42)

# ---------------------------------------------------------------------------
#  OUTPUT DIRECTORY
# ---------------------------------------------------------------------------
OUTDIR = f"/workspace/hologram/outs/stacked_3d_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
#  PHYSICAL & GEOMETRICAL CONSTANTS
# ---------------------------------------------------------------------------
FREQ_HZ            = 0.5e6
SOUND_SPEED_WATER  = DTYPE(1500.0)
SOUND_SPEED_SIL    = DTYPE(1600.0) # https://chatgpt.com/share/68637b75-c3dc-8000-ba17-326069cef557
OMEGA              = DTYPE(2 * np.pi * FREQ_HZ)

BOWL_DIAM_MM, BOWL_ROC_MM = 50, 50.0
FOCAL_DISTANCE_MM, TARGET_RADIUS_MM = 50.0, 0.5
SIDELOBE_RADIUS_MM = 3.0  # Region to suppress sidelobes

NUM_LENSES            = 10
LENS_WIDTH_MM         = 50.0
LENS_THICKNESS_MM     = 0.3
BACKING_THICKNESS_MM  = 0.5
GAP_BETWEEN_LENSES_MM = 0.01
LENS_START_Z_MM       = 3.5

VOXEL_SIZE_MM = 0.4          # controllable resolution in the optimisable layer
DX_MM         = 0.2          # global grid spacing (coarser to save memory)
DX_M          = DTYPE(DX_MM / 1e3)

GRID_SIZE_MM = (50, 50, 65)
N = tuple(int(d / DX_MM) for d in GRID_SIZE_MM)

domain = Domain(N, tuple([DX_M] * 3))
print("Domain voxels:", N)

# ---------------------------------------------------------------------------
#  STATIC PRE‑COMPUTATIONS (all ints, baked into the XLA graph)
# ---------------------------------------------------------------------------
NUM_XY_SEGMENTS       = int(LENS_WIDTH_MM / VOXEL_SIZE_MM)      # 30
TOTAL_VOXELS_PER_LENS = NUM_XY_SEGMENTS ** 2

lens_xy_start_vox   = int(((GRID_SIZE_MM[0] - LENS_WIDTH_MM) / 2) / DX_MM)
voxel_size_in_grid  = int(VOXEL_SIZE_MM / DX_MM)
lens_width_vox      = NUM_XY_SEGMENTS * voxel_size_in_grid
lens_thickness_vox  = int(LENS_THICKNESS_MM / DX_MM)
backing_vox         = int(BACKING_THICKNESS_MM / DX_MM)
gap_vox             = int(GAP_BETWEEN_LENSES_MM / DX_MM)
lens_start_z_vox    = int(LENS_START_Z_MM / DX_MM)

# ---------------------------------------------------------------------------
#  SIDELOBE SUPPRESSION PARAMETERS
# ---------------------------------------------------------------------------
SIDELOBE_PENALTY = 0.5

# ---------------------------------------------------------------------------
#  VISUALISATION PARAMETERS
# ---------------------------------------------------------------------------
VMAX = 1.0

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

    # ── complex phase term (cast to complex64 once) ──────────────────────────
    phase          = -OMEGA / SOUND_SPEED_WATER * R              # float32
    field_complex  = jnp.exp(1j * phase).astype(CDTYPE)          # complex64
    zeros_complex  = jnp.zeros_like(field_complex)               # same dtype

    src = jnp.where(mask, field_complex, zeros_complex)          # complex64
    return FourierSeries(src[..., None], domain)

bowl_source = create_bowl_source_3d()

# ---------------------------------------------------------------------------
#  VECTORISED PARAMS → SOUND‑SPEED FIELD
# ---------------------------------------------------------------------------
@jit
def params_to_sos(lens_params: jnp.ndarray) -> FourierSeries:
    """Convert flat parameter vector to a Fourier‑series sound‑speed field."""
    # (ℓ, segX, segY) in [0,1] after sigmoid
    matfrac = jax.nn.sigmoid(
        lens_params.reshape(NUM_LENSES, NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
    ).astype(DTYPE)

    # Upsample to full XY resolution (repeat, not interp.)
    matfrac = jnp.repeat(jnp.repeat(matfrac, voxel_size_in_grid, 1),
                         voxel_size_in_grid, 2)          # (ℓ, X, Y)

    # Build optimisable + backing layers for each lens
    opt_layer  = jnp.repeat(matfrac[..., None], lens_thickness_vox, -1)
    backing    = jnp.ones_like(opt_layer[..., :backing_vox])
    lens_block = jnp.concatenate([opt_layer, backing], -1)      # silicone=1

    gap_block  = jnp.zeros_like(opt_layer[..., :gap_vox])       # water gap

    # Assemble stack along Z
    blocks = []
    for idx in range(NUM_LENSES):
        blocks.append(lens_block[idx])
        if idx < NUM_LENSES - 1:
            blocks.append(gap_block[idx])                       # same XY size
    stack = jnp.concatenate(blocks, -1)                         # (X,Y,Z_lens)

    # Embed in global grid
    sos = jnp.ones(N, dtype=DTYPE) * SOUND_SPEED_WATER
    sos = sos.at[
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_start_z_vox:lens_start_z_vox + stack.shape[-1],
    ].set((SOUND_SPEED_SIL - SOUND_SPEED_WATER) * stack + SOUND_SPEED_WATER)

    return FourierSeries(sos[..., None], domain)

# ---------------------------------------------------------------------------
#  FIELD SOLVER (cached XLA trace)
# ---------------------------------------------------------------------------
@jit
def compute_field(lens_params: jnp.ndarray):
    medium = Medium(domain=domain,
                    sound_speed=params_to_sos(lens_params),
                    pml_size=10)
    return helmholtz_solver(medium, OMEGA, bowl_source,
                            tol=1e-3, checkpoint=False)        # jwave is jit‑able

# ---------------------------------------------------------------------------
#  OBJECTIVE (mean pressure in target sphere)
# ---------------------------------------------------------------------------
target_mask = None   # cached in host memory

def objective(lens_params: jnp.ndarray):
    global target_mask
    field = compute_field(lens_params).on_grid
    press = jnp.abs(field[..., 0]) if field.ndim == 4 else jnp.abs(field)

    if target_mask is None:
        x = jnp.arange(N[0]) * DX_MM
        y = jnp.arange(N[1]) * DX_MM
        z = jnp.arange(N[2]) * DX_MM
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        target_mask = ((X - GRID_SIZE_MM[0] / 2) ** 2 +
                       (Y - GRID_SIZE_MM[1] / 2) ** 2 +
                       (Z - FOCAL_DISTANCE_MM) ** 2) <= TARGET_RADIUS_MM ** 2
        target_mask = target_mask.astype(DTYPE)

    mean_p = (press * target_mask).sum() / (target_mask.sum() + 1e-8)
    return -mean_p

objective_and_grad = jit(value_and_grad(objective))

# ---------------------------------------------------------------------------
#  HELPER FUNCTIONS FOR SIDELOBE OBJECTIVE
# ---------------------------------------------------------------------------
@jit
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
#  SIDELOBE SUPPRESSION OBJECTIVE
# ---------------------------------------------------------------------------
@jit
def objective_sidelobe(lens_params: jnp.ndarray):
    """Maximize pressure in target area while suppressing sidelobes."""
    field = compute_field(lens_params).on_grid
    
    if field.ndim == 4:
        pressure_magnitude = jnp.abs(field[..., 0])
    else:
        pressure_magnitude = jnp.abs(field)
    
    # Pre-computed masks
    target_mask = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_region = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, SIDELOBE_RADIUS_MM
    ).astype(DTYPE)
    
    sidelobe_mask = sidelobe_region - target_mask
    
    # Vectorized calculations
    target_pressure = jnp.sum(pressure_magnitude * target_mask) / (jnp.sum(target_mask) + 1e-8)
    sidelobe_pressure = jnp.sum(pressure_magnitude * sidelobe_mask) / (jnp.sum(sidelobe_mask) + 1e-8)
    
    return -(target_pressure - SIDELOBE_PENALTY * sidelobe_pressure)

# ---------------------------------------------------------------------------
#  OBJECTIVE SHAPE
def objective_shape(params, amp_target=1.50, alpha=3.0):
    field = compute_field(params).on_grid
    amp   = jnp.abs(field[...,0])
    target_mask = create_spherical_mask_vectorized(
        GRID_SIZE_MM[0]/2, GRID_SIZE_MM[1]/2, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
    ).astype(DTYPE)
    L_in  = jnp.mean((amp - amp_target)**2 * target_mask)
    L_out = jnp.mean(amp**2 * (1 - target_mask))
    # optional: push sigmoid outputs towards {0,1} for binary printability
    #bin_reg = 1e-3 * jnp.mean(jax.nn.sigmoid(params)*(1-jax.nn.sigmoid(params)))
    return L_in + alpha*L_out #+ bin_reg

# ---------------------------------------------------------------------------

objective_sidelobe_and_grad = jit(value_and_grad(objective_sidelobe))
objective_shape_and_grad = jit(value_and_grad(objective_shape))

# ---------------------------------------------------------------------------
#  OPTIMISER
# ---------------------------------------------------------------------------
# Choose objective function

learning_rate = 0.3
optimiser     = optax.adamw(learning_rate)
total_params  = NUM_LENSES * TOTAL_VOXELS_PER_LENS
params        = 0.2 * (random.uniform(key, (total_params,), dtype=DTYPE) - 0.5)
opt_state     = optimiser.init(params)

# Select objective function
selected_objective_and_grad = objective_sidelobe_and_grad

@jit
def optimisation_step(params, opt_state):
    loss, grads = selected_objective_and_grad(params)
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
# ---------------------------------------------------------------------------
#  DEBUG VISUALISATION HELPERS  << NEW >>
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _save_bowl_source_xz(save_path: str):
    """XZ slice (y‑mid) of the complex bowl source magnitude."""
    y_mid = N[1] // 2
    src   = bowl_source.on_grid[..., 0]          # complex field
    mag   = np.abs(np.asarray(src))              # host‑side for matplotlib

    plt.figure(figsize=(6,4))
    plt.imshow(mag[:, y_mid, :].T,
               extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[2], 0],
               cmap='hot', aspect='auto', origin='upper')
    plt.title("Bowl source |p| (XZ slice)")
    plt.xlabel("x [mm]"); plt.ylabel("z [mm]")
    plt.colorbar(label='|p|')
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    print("Saved:", save_path)

def _save_pressure_xz(lens_params, tag):                   # << MOD >>
    """
    Save an X‑Z slice of |p|.
    `tag` can be an int (iteration index) or a string such as “no_lens”.
    """
    y_mid   = N[1] // 2
    field   = compute_field(lens_params).on_grid
    press   = np.abs(np.asarray(field[..., 0] if field.ndim == 4 else field))

    if isinstance(tag, int):
        fname = f"{OUTDIR}/pressure_iter_{tag:03d}.png"
        title = f"|p| after iter {tag:03d}"
    else:
        fname = f"{OUTDIR}/pressure_{tag}.png"
        title = f"|p| – {tag}"

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(press[:, y_mid, :].T,
                   extent=[0, GRID_SIZE_MM[0], GRID_SIZE_MM[2], 0],
                   cmap='hot', aspect='auto', origin='upper', vmin=0, vmax=VMAX)
    ax.set_title(title + "  (XZ slice, y = mid)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    fig.colorbar(im, ax=ax, label='|p|')

    # green dotted boxes .....................................................
    x0_mm = lens_xy_start_vox * DX_MM
    w_mm  = lens_width_vox     * DX_MM
    for i in range(NUM_LENSES):
        z0_vox = lens_start_z_vox + i * (lens_thickness_vox + backing_vox + gap_vox)
        z1_vox = z0_vox + lens_thickness_vox
        rect   = Rectangle((x0_mm, z0_vox * DX_MM),
                           width=w_mm, height=lens_thickness_vox * DX_MM,
                           linewidth=1.2, edgecolor='lime', linestyle=':', fill=False)
        ax.add_patch(rect)

    ax.plot(GRID_SIZE_MM[0]/2, FOCAL_DISTANCE_MM, 'gx', ms=10, mew=2)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()
    print("Saved:", fname)

# ---------------------------------------------------------------------------
#  SOS‑HISTOGRAM HELPER       << NEW >>
# ---------------------------------------------------------------------------
def _save_sos_hist(lens_params, iter_idx):
    """Save a histogram of the sound‑speed voxels in the optimisable region."""
    sos = np.asarray(params_to_sos(lens_params).on_grid[..., 0])

    sos_opt = sos[
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_xy_start_vox:lens_xy_start_vox + lens_width_vox,
        lens_start_z_vox:lens_start_z_vox +                       # full lens stack
        NUM_LENSES * (lens_thickness_vox + backing_vox + gap_vox) - gap_vox
    ].ravel()

    plt.figure(figsize=(5, 4))
    plt.hist(sos_opt, bins=50,
             range=(float(SOUND_SPEED_WATER), float(SOUND_SPEED_SIL)))
    plt.xlabel("Sound speed [m s⁻¹]"); plt.ylabel("Voxel count")
    plt.title(f"sos distribution @ iter {iter_idx:03d}")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/hist_iter_{iter_idx:03d}.png", dpi=200)
    plt.close()
    print("Saved histogram for iter", iter_idx)

# ---------------------------------------------------------------------------
#  PRESSURE‑METRIC HELPER     << NEW >>
# ---------------------------------------------------------------------------
def _target_pressure(lens_params):
    """Return mean |p| in the target sphere (host‑side float)."""
    field = compute_field(lens_params).on_grid
    press = jnp.abs(field[..., 0] if field.ndim == 4 else field)

    global target_mask
    if target_mask is None:                       # create once, reuse later
        x = jnp.arange(N[0]) * DX_MM
        y = jnp.arange(N[1]) * DX_MM
        z = jnp.arange(N[2]) * DX_MM
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        target_mask = (((X - GRID_SIZE_MM[0]/2)**2 +
                        (Y - GRID_SIZE_MM[1]/2)**2 +
                        (Z - FOCAL_DISTANCE_MM)**2) <= TARGET_RADIUS_MM**2).astype(DTYPE)
    return float((press * target_mask).sum() / (target_mask.sum() + 1e-8))

# ---------------------------------------------------------------------------
#  BASELINE (all‑water, “no_lens”)         << NEW >>
# ---------------------------------------------------------------------------
BASELINE_PARAMS = jnp.full((total_params,), -10.0, dtype=DTYPE)   # sigmoid≈0 → water
baseline_field  = compute_field(BASELINE_PARAMS).on_grid
np.save(f"{OUTDIR}/pressure_no_lens.npy",
        np.asarray(baseline_field[..., 0] if baseline_field.ndim == 4 else baseline_field,
                   dtype=np.float32))
_save_pressure_xz(BASELINE_PARAMS, "no_lens")
print("Baseline simulation stored as 'no_lens'.")

baseline_p = _target_pressure(BASELINE_PARAMS)

# ---------------------------------------------------------------------------
#  TRAINING LOOP
# ---------------------------------------------------------------------------
_save_bowl_source_xz(f"{OUTDIR}/bowl_source_xz.png")

N_ITERS          = 30
losses           = []
pressures        = []          # mean |p| in target each iter
t0 = time.time()

for i in range(N_ITERS):
    params, opt_state, loss = optimisation_step(params, opt_state)
    losses.append(float(loss))

    # target pressure metric .................................................
    p_now = _target_pressure(params)
    pressures.append(p_now)

    if i % 5 == 0:
        # percent improvement vs 5 iters before (or baseline for i==0)
        p_prev = baseline_p if i == 0 else pressures[i-5]
        pct    = (p_now - p_prev) / abs(p_prev + 1e-12) * 100.0

        print(f"iter {i:02d} | loss {loss:10.6f} | ⟨|p|⟩ {p_now:.5e}  "
              f"(+{pct:+.1f}% vs {('baseline' if i==0 else f'iter {i-5:02d}')} )")

        _save_pressure_xz(params, i)
        _save_sos_hist(params, i)

print(f"\nFinished in {time.time() - t0:.1f} s")
overall_pct = (pressures[-1] - baseline_p) / abs(baseline_p) * 100.0
print(f"Mean‑pressure improvement: +{overall_pct:.1f}%  (baseline → final)")

# ---------------------------------------------------------------------------
#  VISUALISATION & EXPORT  (unchanged except for dtype casts)
# ---------------------------------------------------------------------------
def save_lens_png(lens_params, fname):
    fig, ax = plt.subplots(1, NUM_LENSES, figsize=(5 * NUM_LENSES, 4))
    if NUM_LENSES == 1:
        ax = [ax]
    for k in range(NUM_LENSES):
        seg = lens_params[k * TOTAL_VOXELS_PER_LENS:(k + 1) * TOTAL_VOXELS_PER_LENS]
        seg = jax.nn.sigmoid(seg).reshape(NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
        ax[k].imshow(seg, vmin=0, vmax=1, cmap='RdBu_r',
                     extent=[0, LENS_WIDTH_MM, 0, LENS_WIDTH_MM])
        ax[k].set_title(f"Lens {k+1}")
        ax[k].set_xlabel("mm"); ax[k].set_ylabel("mm")
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

save_lens_png(params, f"{OUTDIR}/final_lenses.png")

# -- Convergence plot
plt.figure(figsize=(6,4))
plt.plot(losses); plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.grid(True)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/convergence.png", dpi=200); plt.close()

# -- Store parameters & metadata
np.save(f"{OUTDIR}/optimised_params.npy", np.asarray(params, dtype=np.float32))
with open(f"{OUTDIR}/run.json", "w") as fp:
    json.dump({
        # Optimization info
        "n_iters": N_ITERS,
        "losses": losses,
        "time_sec": time.time() - t0,
        "grid_voxels": N,
        "dtype": "float32/bfloat16",
        "objective": "sidelobe_suppression" if USE_SIDELOBE_OBJECTIVE else "standard",
        "sidelobe_radius_mm": SIDELOBE_RADIUS_MM if USE_SIDELOBE_OBJECTIVE else None,
        "sidelobe_penalty": SIDELOBE_PENALTY if USE_SIDELOBE_OBJECTIVE else None,
        
        # Physical parameters - essential for reconstruction
        "physical_params": {
            "num_lenses": NUM_LENSES,
            "lens_width_mm": LENS_WIDTH_MM,
            "lens_thickness_mm": LENS_THICKNESS_MM,
            "backing_thickness_mm": BACKING_THICKNESS_MM,
            "gap_between_lenses_mm": GAP_BETWEEN_LENSES_MM,
            "lens_start_z_mm": LENS_START_Z_MM,
            "voxel_size_mm": VOXEL_SIZE_MM,
            "num_xy_segments": NUM_XY_SEGMENTS,
            "total_voxels_per_lens": TOTAL_VOXELS_PER_LENS,
            "total_params": total_params,
            "dx_mm": DX_MM,
            "grid_size_mm": GRID_SIZE_MM,
            "freq_hz": FREQ_HZ,
            "sound_speed_water": float(SOUND_SPEED_WATER),
            "sound_speed_sil": float(SOUND_SPEED_SIL),
            "bowl_diam_mm": BOWL_DIAM_MM,
            "bowl_roc_mm": BOWL_ROC_MM,
            "focal_distance_mm": FOCAL_DISTANCE_MM,
            "target_radius_mm": TARGET_RADIUS_MM
        }
    }, fp, indent=2)

print("All artefacts stored in:", OUTDIR)

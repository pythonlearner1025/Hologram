from dataclasses import replace
import jax.numpy as jnp
from jwave.geometry import Medium
from jaxdf import Domain, FiniteDifferences
from jwave.acoustics.pml import complex_pml_on_grid
from jwave.acoustics.operators import (
    laplacian_with_pml, wavevector, scale_source_helmholtz, helmholtz
)
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave import FourierSeries
from jax.scipy.sparse.linalg import bicgstab, gmres
import jax

def _conjugate_pml_params(params: dict) -> dict:
    """Return a deep-copied Laplacian-param dict with conjugated PML tensors."""
    new_params = dict(params)                       # shallow copy is enough
    pml_list   = params["pml_on_grid"]
    
    # Handle FourierSeries objects - extract data, conjugate, wrap back
    conjugated_pml = []
    for p in pml_list:
        if hasattr(p, 'on_grid'):  # It's a FourierSeries
            conj_data = jnp.conj(p.on_grid)
            conj_fs = FourierSeries(conj_data, p.domain)
            conjugated_pml.append(conj_fs)
        else:  # It's a plain array
            conjugated_pml.append(jnp.conj(p))
    
    new_params["pml_on_grid"] = conjugated_pml
    return new_params


def build_adjoint_medium(fwd_med: Medium) -> Medium:
    """
    Create a *shallow* clone of `fwd_med`.

    All physical fields (c, ρ, α) stay identical – we'll inject the
    conjugated stretch tensors downstream, so no modification to the
    dataclass itself is required.
    """
    return replace(fwd_med)                         # dataclasses utility


def adjoint_rhs(dl_du: jnp.ndarray, domain) -> FourierSeries:
    """
    Pack ∂L/∂u (already conjugated) into a `FourierSeries` that JWAVE expects.
    """
    if dl_du.ndim == 3:                             # add channel axis if needed
        dl_du = dl_du[..., None]
    return FourierSeries(dl_du, domain)



def helmholtz_solver_precond(medium, omega, source, params, M, **kwargs):
    source = scale_source_helmholtz(source, medium)

    def helm_func(u):
        return helmholtz(u, medium, omega=omega, params=params)

    if kwargs["checkpoint"]:
        helm_func = jax.checkpoint(helm_func)

    guess = source * 0

    tol = kwargs["tol"] if "tol" in kwargs else 1e-3
    restart = kwargs["restart"] if "restart" in kwargs else 10
    maxiter = kwargs["maxiter"] if "maxiter" in kwargs else 1000

    out = gmres(
        helm_func,
        source,
        guess,
        M=M,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        solve_method='batched'
    )[0]

    return -1j * omega * out

def downsample_average(arr):
    shape = arr.shape
    arr_resh = arr.reshape((shape[0]//2, 2, shape[1]//2, 2, shape[2]//2, 2))
    return arr_resh.mean(axis=(1,3,5))

def prolong(coarse):
    sx, sy, sz = coarse.shape
    fine = jnp.zeros((2*sx, 2*sy, 2*sz), dtype=coarse.dtype)
    fine = fine.at[0::2, 0::2, 0::2].set(coarse)
    fine = fine.at[1::2, 0::2, 0::2].set(0.5 * (coarse[:-1,:,:] + coarse[1:,:,:]))
    fine = fine.at[0::2, 1::2, 0::2].set(0.5 * (fine[0::2,0::2,0::2] + fine[0::2,2::2,0::2]))
    fine = fine.at[1::2, 1::2, 0::2].set(0.5 * (fine[1::2,0::2,0::2] + fine[1::2,2::2,0::2]))
    fine = fine.at[0::2, 0::2, 1::2].set(0.5 * (fine[0::2,0::2,0::2] + fine[0::2,0::2,2::2]))
    fine = fine.at[1::2, 0::2, 1::2].set(0.5 * (fine[1::2,0::2,0::2] + fine[1::2,0::2,2::2]))
    fine = fine.at[0::2, 1::2, 1::2].set(0.5 * (fine[0::2,1::2,0::2] + fine[0::2,1::2,2::2]))
    fine = fine.at[1::2, 1::2, 1::2].set(0.5 * (fine[1::2,1::2,0::2] + fine[1::2,1::2,2::2]))
    return fine

def restrict(arr):
    # Simple average restriction (full weighting approximation)
    pad_arr = jnp.pad(arr, ((1,1),(1,1),(1,1)), mode='edge')
    return downsample_average(pad_arr[1:-1,1:-1,1:-1])

def apply_P(level, u):
    domain = level["domain"]
    med = level["med"]
    lapl_params = level["lapl_params"]
    fd_u = FiniteDifferences(u[..., None], domain, accuracy=2)
    h = laplacian_with_pml(fd_u, med, omega=level["omega"], params=lapl_params) + wavevector(fd_u, med, omega=level["omega"])
    return h.on_grid[..., 0]

def relax(level, u, v):
    omega_jac = 0.8  # Damped Jacobi weight; can tune between 0.5-0.9 for stability
    p_u = apply_P(level, u)
    res = v - p_u
    u = u + omega_jac * res / level["D"]
    return u

def v_cycle(levels, level, v, u, nu1=3, nu2=3):
    '''
        Recursive V-cycle for approximating P^{-1} v
        Where P u = v 
        and P is laplace_shifted Helmholtz operator
    '''
    if level == len(levels) - 1:
        # Coarsest level: relax more times for approximate solve
        for _ in range(20):
            u = relax(levels[level], u, v)
        return u
    else:
        # Pre-smoothing
        for _ in range(nu1):
            u = relax(levels[level], u, v)
        # Residual
        res = v - apply_P(levels[level], u)
        # Restrict residual
        res_coarse = restrict(res)
        # Recursive coarse solve
        e_coarse = jnp.zeros_like(res_coarse, dtype=complex)
        e_coarse = v_cycle(levels, level + 1, res_coarse, e_coarse, nu1, nu2)
        # Prolongate correction
        e_fine = prolong(e_coarse)
        u = u + e_fine
        # Post-smoothing
        for _ in range(nu2):
            u = relax(levels[level], u, v)
        return u


def solve_adjoint_precond(dl_du_conj: jnp.ndarray,
                  domain,
                  fwd_medium: Medium,
                  omega: float,
                  tol: float = 1e-3,
                  shift: float = 0.5
                  ):
    """
    Solve H† λ  =  (∂L/∂u)* using GMRES, where H† is the true discrete
    adjoint of JWAVE's PML Helmholtz operator.

    Returns
    -------
    jwave.discretization.FourierSeries
        The adjoint field λ on the grid.
    """
    # 1. Medium copy (same domain & material maps)
    adj_med = build_adjoint_medium(fwd_medium)

    # 2. Laplacian parameter bundle + PML conjugation
    #    JWAVE builds this once per operator; we override the PML only.
    dummy_fs   = adjoint_rhs(dl_du_conj, domain)
    lapl_params = laplacian_with_pml.default_params(dummy_fs,
                                                    adj_med,
                                                    omega=omega)
    lapl_params = _conjugate_pml_params(lapl_params)

    w = omega**2 / adj_med.sound_speed.on_grid[...,0]**2  # True wavevector multiplier

    # True Helmholtz params (for the operator)
    helm_params = {
        "laplacian": lapl_params,
        "wavevector": None  # Forces computation from medium
    }

    # Build multigrid levels for shifted preconditioner (FD-based)
    P_levels = []
    current_med = adj_med
    current_domain = domain
    while min(current_domain.N) > 32:
        lapl_params_level = laplacian_with_pml.default_params(
            FiniteDifferences(jnp.zeros(current_domain.N), current_domain, accuracy=2),
            current_med,
            omega=omega
        )
        lapl_params_level = _conjugate_pml_params(lapl_params_level)
        w_level = omega**2 / current_med.sound_speed.on_grid[...,0]**2
        w_shifted_level = (1 + 1j * shift) * w_level
        dx = current_domain.dx[0]
        D = w_shifted_level - (6 / dx**2)  # Approx diag for 3D FD Laplacian
        current_med_shifted = current_med.replace(
            sound_speed=current_med.sound_speed / jnp.sqrt(1 + 1j * shift)
        )
        P_levels.append({
            "domain": current_domain,
            "med": current_med_shifted,
            "lapl_params": lapl_params_level,
            "D": D,
            "omega": omega
        })
        # Coarsen
        new_N = tuple(n // 2 for n in current_domain.N)
        if min(new_N) <= 32:
            break
        new_dx = tuple(d * 2 for d in current_domain.dx)
        new_domain = Domain(new_N, new_dx)
        k2_fine = omega**2 / current_med.sound_speed.on_grid[...,0]**2
        k2_coarse = downsample_average(k2_fine)
        sos_coarse = omega / jnp.sqrt(k2_coarse + 1e-10)  # Avoid div-by-zero
        new_sound_speed = FourierSeries(sos_coarse[..., None], new_domain)
        current_med = Medium(domain=new_domain, sound_speed=new_sound_speed, pml_size=current_med.pml_size)

    # Preconditioner function
    def preconditioner(fs_v):
        v = fs_v.on_grid[..., 0]
        guess = jnp.zeros_like(v, dtype=complex)
        z = v_cycle(P_levels, 0, v, guess, nu1=3, nu2=3)
        return FourierSeries(z[..., None], domain)

    # 3. GMRES solve – no checkpointing, gradients stop here
    λ = helmholtz_solver_precond(adj_med,
                        omega,
                        dummy_fs,
                        helm_params,
                        preconditioner,
                        tol=tol,
                        checkpoint=False
                        )
    return λ 
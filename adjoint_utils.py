# adjoint_utils.py
# ----------------------------------------------------------------------
# Build & solve the true adjoint Helmholtz problem for JWAVE (PML aware)
# ----------------------------------------------------------------------

from dataclasses import replace
import jax.numpy as jnp
from jwave.geometry import Medium
from jwave.acoustics.pml import complex_pml_on_grid
from jwave.acoustics.operators import laplacian_with_pml, wavevector
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave import FourierSeries


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


def solve_adjoint(dl_du_conj: jnp.ndarray,
                  domain,
                  fwd_medium: Medium,
                  omega: float,
                  tol: float = 1e-3):
    """
    Solve H† λ  =  (∂L/∂u)* using GMRES, where H† is the true discrete
    adjoint of JWAVE's PML Helmholtz operator.

    Returns
    -------
    jwave.discretization.FourierSeries
        The adjoint field λ on the grid.
    """
    # 1. Medium copy (same domain & material maps)
    adj_med = replace(fwd_medium)

    # 2. Laplacian parameter bundle + PML conjugation
    #    JWAVE builds this once per operator; we override the PML only.
    if dl_du_conj.ndim == 3:
        dl_du_conj = dl_du_conj[..., None]
    fs = FourierSeries(dl_du_conj, domain)
    lapl_params = laplacian_with_pml.default_params(fs, adj_med, omega=omega)
    lapl_params = _conjugate_pml_params(lapl_params)

    # Construct full params dict for helmholtz_solver
    params = {
        **lapl_params,
        "wavevector": wavevector.default_params(fs, adj_med, omega=omega)
    }
    
    # 3. GMRES solve – no checkpointing, gradients stop here
    λ = helmholtz_solver(adj_med,
                         omega,
                         fs,
                         tol=tol,
                         checkpoint=False,
                         params=params)   # <-- extra kwarg
    return λ 
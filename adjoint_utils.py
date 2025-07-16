# adjoint_utils.py
# ----------------------------------------------------------------------
# Build & solve the true adjoint Helmholtz problem for JWAVE (PML aware)
# ----------------------------------------------------------------------

from dataclasses import replace
import jax.numpy as jnp
from jwave.geometry import Medium
from jwave.acoustics.pml import complex_pml_on_grid
from jwave.acoustics.operators import laplacian_with_pml, wavevector
from jax.scipy.sparse.linalg import bicgstab
from jwave import FourierSeries


def _conjugate_pml_params(params: dict) -> dict:
    """
        Return a deep-copied Laplacian-param dict with conjugated PML tensors. 
        PML works by 'stretching' coordinate x into complex one x_bar = x + i \int_0^x \sigma{\xi} d\xi   
        where \sig(x) \geq is a dampening profile. In fwd this dampens outgoing waves, 
        in bwd it dampens incoming sensitivities. 
    """
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

def solve_adjoint(dl_du_conj: jnp.ndarray,
                  domain,
                  fwd_medium: Medium,
                  omega: float,
                  tol: float = 1e-6):
    """
    Solve H† λ  =  (∂L/∂u)* using Bi-CGSTAB, where H† is the true discrete
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

    # Construct full params dict
    params = {
        **lapl_params,
        "wavevector": wavevector.default_params(fs, adj_med, omega=omega)
    }
    
    # 3. Build matrix-free Helmholtz operator for Bi-CGSTAB
    def helm(v: jnp.ndarray) -> jnp.ndarray:
        """
        Apply H† to a complex grid-field `v` and return a same-shaped array.
        * `v` : (Nx, Ny, Nz) complex64
        * Returns : H† v   (same dtype/shape)
        """
        v_fs = FourierSeries(v[..., None], domain)          # wrap for JWAVE
        Av   = laplacian_with_pml(v_fs, adj_med, omega=omega,
                                  params=params).on_grid[..., 0]
        # Add k²(x) term
        k2   = params["wavevector"]
        if k2.ndim == 4:
            k2 = k2[..., 0]
        Av  += k2 * v
        return Av

    # 4. Diagonal (Jacobi) preconditioner
    dx   = domain.dx[0]                                         # cubic voxels
    diag = (-6.0 / dx**2) + params["wavevector"]
    if diag.ndim == 4:                                          # drop channel
        diag = diag[..., 0]
    inv_diag = jnp.where(diag != 0, 1.0 / diag, 0.0).astype(jnp.float32)
    M = lambda r: inv_diag * r                                  # r / diag(A)

    b  = fs.on_grid[..., 0]                                     # right-hand side
    x0 = jnp.zeros_like(b)

    λ_grid, info = bicgstab(helm, b,
                            x0=x0,
                            M=M,
                            tol=tol,
                            maxiter=400)

    if info != 0:
        raise RuntimeError(f"Bi-CGSTAB failed: info={info}")

    λ = FourierSeries(λ_grid[..., None], domain)
    return λ 
from typing import Callable
from jaxdf.operators import gradient
from jaxdf import FourierSeries

from jwave.geometry import Medium, Domain
from jwave import CDTYPE  # Assuming CDTYPE is defined elsewhere, e.g., jnp.complex64

from .pml import complex_pml_on_grid  # Assuming pml.py is in the same directory

import jax.numpy as jnp

def shifted_laplacian_with_pml(omega: float, medium: Medium, shift: float) -> Callable[[FourierSeries], FourierSeries]:
    """Construct shifted Laplacian preconditioner with proper PML handling."""
    
    # Extract PML parameters from your existing setup
    dummy_field = FourierSeries(jnp.ones((*medium.domain.N, 1), dtype=CDTYPE), medium.domain)
    
    # Compute the PML factors (1/s_j for each direction j) on the grid
    pml = complex_pml_on_grid(medium, omega)
    
    # Compute the shifted k^2 term
    # Handle if sound_speed is a field or scalar
    if isinstance(medium.sound_speed, Field):
        c = medium.sound_speed.on_grid
    else:
        c = jnp.asarray(medium.sound_speed)
    k_sq = (omega ** 2 / c ** 2)
    shifted_k_sq = k_sq * (1 + 1j * shift)
    # If scalar, broadcast; if array, add channel dim later
    
    def preconditioner_operator(u: FourierSeries) -> FourierSeries:
        # Initialize stretched Laplacian as zero field
        stretched_lap = dummy_field * 0j
        
        # Loop over dimensions to compute sum_j (1/s_j) * partial_j ( (1/s_j) * partial_j u )
        ndim = medium.domain.ndim
        for dim in range(ndim):
            # Create field for 1/s_j in this direction
            one_s_arr = pml[..., dim][..., None]  # Shape: (*N, 1)
            one_s_field = FourierSeries(one_s_arr, medium.domain)
            
            # Compute partial derivative in direction dim
            grad_dim = gradient(u)[dim]
            
            # Multiply by 1/s_j
            inner = one_s_field * grad_dim
            
            # Compute partial derivative of inner in direction dim
            grad_dim_inner = gradient(inner)[dim]
            
            # Multiply by 1/s_j again
            term = one_s_field * grad_dim_inner
            
            # Add to total
            stretched_lap += term
        
        # Compute mass term: shifted_k_sq * u
        if jnp.ndim(shifted_k_sq) == 0:
            mass_term = shifted_k_sq * u
        else:
            shifted_k_sq_field = FourierSeries(shifted_k_sq[..., None], medium.domain)
            mass_term = shifted_k_sq_field * u
        
        # P u = - stretched_lap - mass_term (since A = -Δ - k^2, P = -Δ~ - (1+iβ)k^2)
        return -stretched_lap - mass_term
    
    return preconditioner_operator


def setup_multigrid_discretization(domain, omega, medium, ppw=6):
    levels = []
    current_n = np.array(medium.domain.N)
    dx = domain.dx[0]

    # Level 0 (finest): 200×200×288, dx=0.125mm
    levels.append({
        'N': current_n,
        'dx': dx,
        'ppw': ppw,  # Your current PPW
        'stencil': 'standard'
    })
    
    # Coarsen by factor of 2 until minimum size
    while np.min(current_n) > 16:
        current_n = current_n // 2
        dx *= 2
        
        # Check points per wavelength at this level
        c_avg = 2000  # Average sound speed
        wavelength = c_avg / (omega / (2 * np.pi))
        ppw = wavelength / dx
        
        # Modified discretization for coarse grids
        if ppw < 4:
            stencil = 'compact_high_order'  # 4th order compact
        else:
            stencil = 'standard'
        
        levels.append({
            'N': current_n.copy(),
            'dx': dx,
            'ppw': ppw,
            'stencil': stencil
        })
    
    return levels
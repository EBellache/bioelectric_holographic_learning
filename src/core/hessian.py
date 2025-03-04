# hessian.py
import jax
import jax.numpy as jnp
from .free_energy import free_energy


def compute_hessian(
    psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg=1.0
):
    """Compute Hessian explicitly of the free-energy."""
    hess_fn = jax.hessian(free_energy)
    hessian_matrix = hessian_fn(
        psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg
    )
    return hessian_matrix


def analyze_stability(hessian_matrix):
    """Explicitly compute eigenvalues to assess goal stability."""
    eigvals = jnp.linalg.eigvalsh(hessian_matrix)
    stability = jnp.all(eigvals > 0)
    return eigvals, stability

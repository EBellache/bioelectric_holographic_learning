# gradient.py
import jax
from .free_energy import free_energy


def compute_gradient(
    psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg=1.0
):
    """Compute the gradient of free-energy explicitly."""
    grad_fn = jax.grad(free_energy, argnums=(0, 2, 4))
    grad_psi, grad_gene_expr, grad_control = grad_fn(
        psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg
    )
    return grad_psi, grad_gene_expr, grad_control

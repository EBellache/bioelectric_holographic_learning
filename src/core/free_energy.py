# free_energy.py
import jax.numpy as jnp


def free_energy(
    psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg=1.0
):
    """Compute the explicit bioelectric-epigenetic free-energy functional."""
    bioelectric_loss = jnp.linalg.norm(psi - psi_target) ** 2
    epigenetic_loss = jnp.linalg.norm(gene_expr - gene_expr_obs) ** 2
    control_cost = lambda_reg * jnp.linalg.norm(control_input) ** 2
    return bioelectric_loss + gene_expr_loss + control_cost

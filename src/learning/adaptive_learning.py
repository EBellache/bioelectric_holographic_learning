# adaptive_learning.py
import jax
import jax.numpy as jnp
from src.core.free_energy import free_energy


def adaptive_parameter_update(
    psi, gene_expr, gene_expr_obs, params, learning_rate=0.01
):
    """Explicitly update PDE parameters via gradient descent."""

    def loss_fn(theta):
        predicted_gene_expr = model(psi, theta)
        return jnp.linalg.norm(predicted_gene_expr - gene_expr_obs) ** 2

    gradients = jax.grad(loss_fn)(theta)
    theta_updated = {k: params[k] - learning_rate * gradients[k] for k in theta.keys()}

    return theta_updated

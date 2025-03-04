# optimal_control.py
import jax
import jax.numpy as jnp
from src.core.free_energy import free_energy


def optimal_control_step(
    psi,
    psi_target,
    gene_expr,
    gene_expr_obs,
    control_input,
    learning_rate=0.01,
    lambda_reg=1.0,
):
    """
    Perform an explicit optimal control update step based on free-energy gradients.
    """
    # Compute gradient explicitly
    grad_fn = jax.grad(free_energy, argnums=4)
    grad_control = grad_fn(
        psi, psi_target, gene_expr, gene_expr_obs, control_input, lambda_reg
    )

    # Explicit update of control input
    updated_control = control_input - learning_rate * grad_control
    return updated_control


def run_optimal_control(
    psi_init,
    psi_target,
    gene_expr_init,
    gene_expr_obs,
    control_init,
    num_steps=100,
    learning_rate=0.01,
    lambda_reg=1.0,
):
    """
    Explicitly run the optimal control loop for a given number of steps.
    """
    control_input = control_init
    psi = psi_init
    gene_expr = gene_expr_init

    for step in range(num_steps):
        control_input = optimal_control_step(
            psi,
            psi_target,
            gene_expr,
            gene_expr_obs,
            control_input,
            learning_rate,
            lambda_reg,
        )

    return control_input

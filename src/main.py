import jax
import jax.numpy as jnp
from src.data.data_structures import PDEState
from src.control.cost_functions import final_state_l2_loss
from src.control.optimization import optimize_control, solve_cable_pde


def main():
    # PDE config
    L = 1.0
    Nx = 128
    dx = L / (Nx - 1)
    T = 1.0
    Nt = 100
    dt = T / Nt
    D = 0.01

    # Initial condition
    x_grid = jnp.linspace(0, L, Nx)
    u_init = jnp.exp(-50 * (x_grid - 0.3) ** 2)

    # Example target for final state
    target = 0.5 * jnp.sin(2 * jnp.pi * x_grid)

    # Wrap the cost function that uses final_state_l2_loss
    def my_cost_func(u_ts: jnp.ndarray) -> float:
        return final_state_l2_loss(u_ts, target)

    # Optimize
    control_2d_opt, cost_hist = optimize_control(
        u_init, Nx, Nt, dt, dx, D, my_cost_func, max_iter=200, lr=1e-2
    )

    # Solve PDE with best control
    u_ts_opt = solve_cable_pde(u_init, control_2d_opt, dt, dx, D)

    print("Optimization done. Final cost:", cost_hist[-1])


if __name__ == "__main__":
    main()

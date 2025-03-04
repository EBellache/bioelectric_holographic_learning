# heisenberg_solver.py
import jax
import jax.numpy as jnp


def heisenberg_step(psi, U_bio, dx, dt, hbar=1.0, m=1.0):
    """Perform a single time step explicitly solving the Heisenberg Schrödinger PDE."""
    laplacian = (jnp.roll(psi, -1) - 2 * psi + jnp.roll(psi, 1)) / dx**2
    dpsi_dt = (-1j / hbar) * (-(hbar**2 / (2 * m)) * laplacian + U_bio * psi)
    psi_next = psi + dt * dpsi_dt  # Forward Euler
    return psi_next


def evolve_heisenberg(psi_init, U_bio_fn, x_grid, t_grid):
    """Explicitly evolve bioelectric wavefunction psi over given time grid."""
    psi_t_series = [psi_init]
    psi_current = psi_init
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    for t in t_grid[:-1]:
        U_bio_current = U_bio_fn(x_grid, t)
        psi_next = heisenberg_step(psi_current, U_bio_current, dx, dt)
        psi_t_series.append(psi_next)
        psi_current = psi_next

    return jnp.array(psi_t_series)

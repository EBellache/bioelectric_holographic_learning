"""
cable_equation.py

Defines a basic 1D cable-equation PDE in continuous form, plus
a simple config dataclass. This does not solve the PDE itself;
see solver_crank_nicolson.py for the actual time-stepping code.
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class CableConfig:
    """
    Holds PDE parameters and grid info for the cable equation.
    """

    L: float = 1.0  # domain length in x
    Nx: int = 100  # number of spatial points
    T: float = 1.0  # final time
    Nt: int = 100  # number of time steps
    D: float = 0.01  # diffusion coefficient
    bc_type: str = "neumann"  # boundary condition, e.g. "neumann" or "dirichlet"
    dt: float = None  # computed from T/Nt if not set

    def __post_init__(self):
        if self.dt is None:
            self.dt = self.T / self.Nt


def init_domain(config: CableConfig):
    """
    Returns the x_grid, dx for the 1D domain [0, L].
    """
    x_grid = jnp.linspace(0, config.L, config.Nx)
    dx = (config.L / (config.Nx - 1)) if config.Nx > 1 else config.L
    return x_grid, dx


def initial_condition(x_grid: jnp.ndarray) -> jnp.ndarray:
    """
    Example: a Gaussian bump as initial voltage.
    Replace with any custom function for cryptic worm initialization.
    """
    return jnp.exp(-50.0 * (x_grid - 0.3) ** 2)

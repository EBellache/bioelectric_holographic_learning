"""
solver_crank_nicolson.py

Implements a Crank–Nicolson time-stepping for the 1D cable equation
using JAX for auto-diff or HPC acceleration.
"""

import jax
import jax.numpy as jnp

from .solver_utils import build_tridiag, apply_neumann_bc, apply_dirichlet_bc
from .cable_equation import CableConfig, init_domain


def crank_nicolson_step(
    u_n: jnp.ndarray,
    ctrl_n: jnp.ndarray,
    ctrl_np1: jnp.ndarray,
    dt: float,
    dx: float,
    D: float,
    bc_type: str = "neumann",
) -> jnp.ndarray:
    """
    One Crank–Nicolson step for the PDE:
      du/dt = D d2u/dx^2 + control(x,t).
    We do:
      (I + r/2 * Lap) * u_{n+1} = (I - r/2 * Lap)*u_n + dt/2*(ctrl_n + ctrl_np1)
    where r = D * dt / dx^2.
    Returns u_{n+1}.
    """
    Nx = u_n.shape[0]
    r = (D * dt) / (dx**2)

    main_diag_A = jnp.ones(Nx) + r
    off_diag_A = -0.5 * r

    main_diag_B = jnp.ones(Nx) - r
    off_diag_B = 0.5 * r

    A_mat = build_tridiag(main_diag_A, off_diag_A, Nx)
    B_mat = build_tridiag(main_diag_B, off_diag_B, Nx)

    rhs = B_mat @ u_n + 0.5 * dt * (ctrl_n + ctrl_np1)

    # Solve for u_np1
    u_np1 = jnp.linalg.solve(A_mat, rhs)

    # Apply boundary conditions. If bc=neumann, we do the naive approach.
    if bc_type == "neumann":
        u_np1 = apply_neumann_bc(u_np1)
    elif bc_type == "dirichlet":
        u_np1 = apply_dirichlet_bc(u_np1)
    return u_np1


def solve_cable_crank_nicolson(
    u_init: jnp.ndarray, control_2d: jnp.ndarray, config: CableConfig
) -> jnp.ndarray:
    """
    Solve the cable PDE in time using Crank–Nicolson.
    control_2d: shape (Nt+1, Nx).
    returns u_ts: shape (Nt+1, Nx).
    """
    dt = config.dt
    Nx = config.Nx
    Nt = config.Nt
    dx = config.L / (Nx - 1)
    bc_type = config.bc_type

    # We store all time steps in a (Nt+1, Nx) array
    u_ts = jnp.zeros((Nt + 1, Nx))
    u_ts = u_ts.at[0, :].set(u_init)

    def time_step_fun(carry, n):
        u_n = carry
        ctrl_n = control_2d[n]
        ctrl_np1 = control_2d[n + 1]
        u_np1 = crank_nicolson_step(u_n, ctrl_n, ctrl_np1, dt, dx, config.D, bc_type)
        return u_np1, u_np1

    indices = jnp.arange(Nt)
    final, all_sol = jax.lax.scan(time_step_fun, u_init, indices)
    u_ts = u_ts.at[1:].set(all_sol)
    return u_ts

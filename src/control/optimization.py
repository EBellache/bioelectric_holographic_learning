"""
optimization.py

Implements PDE solver (Crank–Nicolson)
+ an optimization loop for PDE control in JAX.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional


# -----------------------------
# PDE Solver: Crank–Nicolson
# -----------------------------
def crank_nicolson_step(
    u_n: jnp.ndarray,
    ctrl_n: jnp.ndarray,
    ctrl_np1: jnp.ndarray,
    dt: float,
    dx: float,
    D: float,
) -> jnp.ndarray:
    """
    Single step of the 1D PDE:
      du/dt = D d2u/dx^2 + control

    Using Crank–Nicolson discretization.
    This is a minimal example (no boundary logic).
    """

    Nx = u_n.shape[0]
    r = D * dt / (dx * dx)

    # Build tri-di matrix A, B for CN:
    # A = (I + r/2 * Lap), B = (I - r/2 * Lap)
    # We'll do naive approach: dense Nx x Nx. For HPC, you'd do a banded or iterative solver.

    diag_main_A = jnp.ones(Nx) + r
    diag_off_A = -0.5 * r
    diag_main_B = jnp.ones(Nx) - r
    diag_off_B = 0.5 * r

    A = build_tridiag(diag_main_A, diag_off_A, Nx)
    B = build_tridiag(diag_main_B, diag_off_B, Nx)

    rhs = B @ u_n + dt * 0.5 * (ctrl_n + ctrl_np1)
    u_np1 = jnp.linalg.solve(A, rhs)
    return u_np1


def build_tridiag(main_diag: jnp.ndarray, off_diag: float, Nx: int) -> jnp.ndarray:
    """
    Construct Nx x Nx tridiagonal matrix with
    main_diag[i] on diagonal, off_diag on sub & super diag.
    """
    mat = jnp.zeros((Nx, Nx))

    def row_fun(i, mat_):
        row = mat_[i]
        row = row.at[i].set(main_diag[i])
        if i > 0:
            row = row.at[i - 1].set(off_diag)
        if i < Nx - 1:
            row = row.at[i + 1].set(off_diag)
        return mat_.at[i].set(row)

    mat = jax.lax.fori_loop(0, Nx, row_fun, mat)
    return mat


def solve_cable_pde(
    u_init: jnp.ndarray, control_2d: jnp.ndarray, dt: float, dx: float, D: float
) -> jnp.ndarray:
    """
    Full time stepping.
    control_2d shape: (Nt+1, Nx). returns (Nt+1, Nx) PDE solution.
    """
    Nt = control_2d.shape[0] - 1
    Nx = u_init.shape[0]
    u_ts = jnp.zeros((Nt + 1, Nx))
    u_ts = u_ts.at[0, :].set(u_init)

    def step_fun(carry, n):
        u_n = carry
        ctrl_n = control_2d[n]
        ctrl_np1 = control_2d[n + 1]
        u_np1 = crank_nicolson_step(u_n, ctrl_n, ctrl_np1, dt, dx, D)
        return u_np1, u_np1

    indices = jnp.arange(Nt)
    final, all_sol = jax.lax.scan(step_fun, u_init, indices)
    u_ts = u_ts.at[1:].set(all_sol)
    return u_ts


# -----------------------------
# Optimization Loop
# -----------------------------
def cost_fn(
    control_flat: jnp.ndarray,
    u_init: jnp.ndarray,
    dt: float,
    dx: float,
    D: float,
    Nx: int,
    Nt: int,
    cost_func: Callable[[jnp.ndarray], float],
) -> float:
    """
    Generic cost function:
    1) Reshape control => (Nt+1, Nx)
    2) Solve PDE => (Nt+1, Nx)
    3) Evaluate cost_func(u_ts)
    plus small reg on control
    """
    control_2d = control_flat.reshape((Nt + 1, Nx))
    u_ts = solve_cable_pde(u_init, control_2d, dt, dx, D)
    cost_val = cost_func(u_ts)
    reg = 1e-3 * jnp.sum(control_2d**2)
    return cost_val + reg


@jax.jit
def gradient_descent(control_flat, step_size, *args):
    # *args => pass to cost_fn
    cval, grad_val = jax.value_and_grad(cost_fn)(control_flat, *args)
    new_control = control_flat - step_size * grad_val
    return new_control, cval


def optimize_control(
    u_init: jnp.ndarray,
    Nx: int,
    Nt: int,
    dt: float,
    dx: float,
    D: float,
    cost_func: Callable[[jnp.ndarray], float],
    max_iter=100,
    lr=1e-2,
):
    """
    Simple gradient descent with JAX for PDE-based control
    """
    # initial guess
    control_init_2d = jnp.zeros((Nt + 1, Nx))
    control_flat = control_init_2d.ravel()
    cost_history = []

    for i in range(max_iter):
        control_flat, cval = gradient_descent(
            control_flat, lr, u_init, dt, dx, D, Nx, Nt, cost_func
        )
        if i % 10 == 0:
            cost_history.append(float(cval))
            print(f"Iter={i}, cost={cval:.6f}")

    return control_flat.reshape((Nt + 1, Nx)), jnp.array(cost_history)

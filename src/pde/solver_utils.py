"""
solver_utils.py

Generic utilities for PDE solvers, including
tridiagonal matrix construction or boundary condition helpers.
"""

import jax
import jax.numpy as jnp


def build_tridiag(main_diag: jnp.ndarray, off_diag: float, Nx: int) -> jnp.ndarray:
    """
    Constructs an Nx x Nx tridiagonal matrix with main_diag[i] on the diagonal
    and 'off_diag' on sub and super diagonals. Very naive approach for demonstration.
    """
    mat = jnp.zeros((Nx, Nx))

    def fill_row(i, mat_):
        row = mat_[i]
        row = row.at[i].set(main_diag[i])
        if i > 0:
            row = row.at[i - 1].set(off_diag)
        if i < Nx - 1:
            row = row.at[i + 1].set(off_diag)
        return mat_.at[i].set(row)

    mat = jax.lax.fori_loop(0, Nx, fill_row, mat)
    return mat


def apply_neumann_bc(u: jnp.ndarray):
    """
    Enforces naive zero-flux (Neumann) boundary conditions by copying neighbor values.
    i.e. derivative=0 => u[0]=u[1], u[-1]=u[-2].
    """
    Nx = u.shape[0]
    # For demonstration:
    u = u.at[0].set(u[1])
    u = u.at[Nx - 1].set(u[Nx - 2])
    return u


def apply_dirichlet_bc(u: jnp.ndarray, left_val=0.0, right_val=0.0):
    """
    Dirichlet boundary: fix u[0] and u[-1] to specified values.
    """
    Nx = u.shape[0]
    u = u.at[0].set(left_val)
    u = u.at[Nx - 1].set(right_val)
    return u

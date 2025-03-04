"""
data_structures.py

Defines custom data classes or named tuples to store
bioelectric waveforms, PDE states, etc.
"""

from typing import NamedTuple
import jax.numpy as jnp


class WormSignal(NamedTuple):
    """Stores a 1D worm signal for time or space indexing."""

    values: jnp.ndarray
    label: str


class PDEState(NamedTuple):
    """Stores PDE solution arrays for multi-time or multi-space data."""

    u_ts: jnp.ndarray  # shape (Nt+1, Nx)
    x_grid: jnp.ndarray  # shape (Nx,)
    t_grid: jnp.ndarray  # shape (Nt+1,)


class ControlField(NamedTuple):
    """Stores a spatiotemporal control array and references to grid."""

    control_2d: jnp.ndarray  # shape (Nt+1, Nx) or (Nx, Nt+1)
    x_grid: jnp.ndarray
    t_grid: jnp.ndarray

"""
cost_functions.py

Defines cost or objective functions for PDE-based optimization.
Including wavelet mismatch or simple L2 difference.
"""

import jax
import jax.numpy as jnp


def final_state_l2_loss(u_ts: jnp.ndarray, u_target: jnp.ndarray) -> float:
    """
    L2 difference between PDE final state and target.
    u_ts: shape (Nt+1, Nx)
    u_target: shape (Nx,)
    returns scalar cost
    """
    u_final = u_ts[-1, :]
    diff = u_final - u_target
    return jnp.sum(diff * diff)


def wavelet_spectral_loss(
    u_ts: jnp.ndarray, wavelet_fn, wavelet_target: jnp.ndarray
) -> float:
    """
    Example wavelet-based mismatch.
    wavelet_fn is a function that transforms a 1D array -> wavelet array.
    wavelet_target is the precomputed wavelet array for normal worm.

    We'll do wavelet on final state for demonstration.
    """
    u_final = u_ts[-1, :]
    wavelet_u = wavelet_fn(u_final)
    # shape-check with wavelet_target
    min_scales = jnp.minimum(wavelet_u.shape[0], wavelet_target.shape[0])
    min_t = jnp.minimum(wavelet_u.shape[1], wavelet_target.shape[1])
    wavelet_u_cut = wavelet_u[:min_scales, :min_t]
    wavelet_t_cut = wavelet_target[:min_scales, :min_t]

    return jnp.sum((wavelet_u_cut - wavelet_t_cut) ** 2)

# quantum_states.py
import jax.numpy as jnp


def initialize_gaussian(x_grid, center, width):
    """Explicitly initialize quantum state as Gaussian distribution."""
    psi_init = jnp.exp(-((x_grid - center) ** 2) / (2 * width**2))
    return normalize_wavefunction(psi_init)


def normalize_wavefunction(psi):
    """Explicit normalization of quantum wavefunction."""
    norm = jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
    return psi / norm

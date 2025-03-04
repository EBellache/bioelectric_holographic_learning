# entropy.py
import jax.numpy as jnp


def shannon_entropy(signal, eps=1e-12):
    """Compute explicit Shannon entropy of a normalized signal."""
    signal_prob = signal / (jnp.sum(signal) + eps)
    entropy = -jnp.sum(signal_prob * jnp.log(signal_prob + eps))
    return entropy


def spectral_entropy(spectral_data):
    """Compute explicit entropy over spectral decomposition data."""
    # Normalize spectral data explicitly
    normalized_data = spectral_data / jnp.sum(spectral_data, axis=0, keepdims=True)
    # Compute entropy explicitly for each timepoint
    entropies = -jnp.sum(normalized_data * jnp.log(normalized_data + 1e-12), axis=0)
    return entropies

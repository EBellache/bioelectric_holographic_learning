# variational_autoencoder.py
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import relu, sigmoid


def encoder(params, x):
    """Explicit encoder mapping inputs to latent self-representation."""
    W_enc, b_enc = params["W_enc"], params["b_enc"]
    z_mean = jnp.dot(x, W_enc) + b_enc
    z_logvar = jnp.dot(x, W_enc) + b_enc  # Simplified for explicit debugging
    return z_mean, z_logvar


def sample_latent(key, z_mean, z_logvar):
    """Explicit reparameterization trick."""
    epsilon = jax.random.normal(key, shape=z_mean.shape)
    return z_mean + jnp.exp(0.5 * z_logvar) * epsilon


def decode(z, params):
    W_dec, b_dec = params["W_dec"], params["b_dec"]
    reconstruction = jnp.dot(z, W_dec) + b_dec
    return reconstruction


def vae_loss(params, x, key):
    z_mean, z_logvar = encoder(x, params)
    epsilon = jax.random.normal(key, shape=z_mean.shape)
    z = z_mean + jnp.exp(0.5 * z_logvar) * epsilon
    x_hat = decode(z, params)

    reconstruction_loss = jnp.sum((x - reconstruction) ** 2)
    kl_loss = -0.5 * jnp.sum(1 + z_logvar - z_mean**2 - jnp.exp(z_logvar))
    return reconstruction_loss + kl_loss

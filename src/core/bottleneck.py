# bottleneck.py
import jax.numpy as jnp
from jax.nn import sigmoid


def bottleneck_encoder(inputs, W_enc, b_enc):
    """Explicitly encode inputs into latent (self) representations."""
    return sigmoid(jnp.dot(W_enc, inputs) + b_enc)


def bottleneck_decoder(latent, W_dec, b_dec):
    """Decode latent self representations explicitly back to original space."""
    return sigmoid(jnp.dot(W_dec, latent) + b_dec)


def representation_loss(inputs, W_enc, b_enc, W_dec, b_dec):
    """Compute explicit reconstruction error (self-model quality)."""
    latent = bottleneck_encoder(inputs, W_enc, b_enc)
    reconstruction = bottleneck_decoder(latent, W_dec, b_dec)
    return jnp.linalg.norm(inputs - reconstruction) ** 2

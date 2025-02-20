import jax.numpy as jnp
import scipy.signal

# Compute Fourier Transform of a signal
def compute_fft(signal):
    return jnp.fft.fft(signal)

# Compute Inverse Fourier Transform
def compute_ifft(signal_fft):
    return jnp.fft.ifft(signal_fft)


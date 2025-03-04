# spectral.py
import jax.numpy as jnp
from scipy.signal import cwt, morlet2
import numpy as np


def spatial_fourier_decomposition(signal_x):
    """Explicit spatial decomposition using Fourier transform."""
    return jnp.fft.fft(signal_x, axis=0)


def gabor_wavelet_transform(signal_t, dt, frequencies):
    """Perform explicit Gabor (Morlet) wavelet transform."""
    widths = frequencies / dt
    spectrogram = cwt(signal_t, morlet2, widths, w=6)
    return jnp.abs(spectrogram)


def spectral_decomposition(signal_xt, dt, frequencies):
    """Combined spatial and temporal spectral decomposition explicitly."""
    spatial_spectrum = spatial_fourier_decomposition(signal_xt)
    # For simplicity, take a representative spatial mode (e.g., mode 0)
    temporal_spectrum = gabor_wavelet_transform(spatial_spectrum[0, :], dt, frequencies)
    return temporal_spectrum

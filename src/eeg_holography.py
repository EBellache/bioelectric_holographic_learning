import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt

# Define EEG-based wave propagation using a Fourier representation
def eeg_wave_function(x, t, params):
    A, k, omega, phi = params
    return A * jnp.sin(k * x - omega * t + phi)

# Compute gradient (simulating neural signal propagation speed)
grad_eeg_wave = jit(grad(eeg_wave_function, argnums=0))

# Sample parameters for testing
params = (1.0, 2.0, 1.0, 0.5)
x_values = jnp.linspace(-10, 10, 500)
t_value = 0.0

# Compute function values
y_values = eeg_wave_function(x_values, t_value, params)
grad_y_values = grad_eeg_wave(x_values, t_value, params)

# Plot results
plt.figure(figsize=(8,4))
plt.plot(x_values, y_values, label="EEG Wave Function")
plt.plot(x_values, grad_y_values, linestyle="dashed", label="Gradient (Signal Propagation)")
plt.legend()
plt.title("EEG-Based Holographic Wave Simulation")
plt.xlabel("x")
plt.ylabel("EEG Signal Intensity")
plt.show()

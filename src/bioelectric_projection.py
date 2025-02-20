import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt

# Define a simple projection function (to be extended in later models)
def projection_function(x, t, params):
    A, k, phi = params
    return A * jnp.sin(k * x + phi * t)

# Compute gradient (as a placeholder for future evolution equations)
grad_projection = jit(grad(projection_function, argnums=0))

# Sample parameters for testing
params = (1.0, 2.0, 0.5)
x_values = jnp.linspace(-10, 10, 500)
t_value = 0.0

# Compute function values
y_values = projection_function(x_values, t_value, params)
grad_y_values = grad_projection(x_values, t_value, params)

# Plot results
plt.figure(figsize=(8,4))
plt.plot(x_values, y_values, label="Projection Function")
plt.plot(x_values, grad_y_values, linestyle="dashed", label="Gradient")
plt.legend()
plt.title("Basic Holographic Projection Simulation")
plt.xlabel("x")
plt.ylabel("Projection Intensity")
plt.show()

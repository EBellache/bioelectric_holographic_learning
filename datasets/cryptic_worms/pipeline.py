
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import cwt, morlet2
from scipy.optimize import minimize
from jax import grad, jit

# ✅ 1. Load & Preprocess Data (Cryptic Worm Dataset)
def load_data(file_path):
    """Load cryptic worm dataset and organize into groups."""
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    groups = {
        "Control": df[df["xfact"] == "(1) Control (normalized)"]["yvar"].values,
        "Nigericin 1wk": df[df["xfact"] == "(3) Nigericin 1wk"]["yvar"].values,
        "Nigericin 3wk": df[df["xfact"] == "(4) Nigericin 3wk"]["yvar"].values,
    }
    return groups

# ✅ 2. Perform Fibonacci-Gabor Spectral Decomposition
def fibonacci_sequence(n_terms):
    """Generate Fibonacci-based frequency scaling."""
    seq = [1, 1]
    for _ in range(2, n_terms):
        seq.append(seq[-1] + seq[-2])
    return np.array(seq) / max(seq)  # Normalize

def spectral_analysis(groups):
    """Perform Gabor-based spectral decomposition on each experimental group."""
    fib_frequencies = fibonacci_sequence(10)
    spectrograms = {group: np.abs(cwt(data, morlet2, fib_frequencies, w=6)) for group, data in groups.items()}
    return spectrograms

# ✅ 3. Clean Spectrograms (Fix NaNs & Normalize)
def clean_spectrograms(spectrograms):
    """Clean and normalize spectrogram data."""
    cleaned_spectrograms = {}
    for group, spec in spectrograms.items():
        spec = np.nan_to_num(spec, nan=np.nanmean(spec))  # Replace NaNs
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))  # Normalize
        cleaned_spectrograms[group] = spec
    return cleaned_spectrograms

# ✅ 4. Define Schrödinger-Cable PDE Solver
def laplacian(psi, dx):
    """Compute discrete Laplacian with Neumann BC."""
    laplacian = jnp.zeros_like(psi)
    laplacian = laplacian.at[1:-1].set((psi[:-2] - 2 * psi[1:-1] + psi[2:]) / dx**2)
    laplacian = laplacian.at[0].set((psi[1] - psi[0]) / dx**2)
    laplacian = laplacian.at[-1].set((psi[-2] - psi[-1]) / dx**2)
    return laplacian

@jit
def schrodinger_cable_step(psi, U_bio, dx, dt, D=1.0, m=1.0, gamma=0.1, psi_rest=0):
    """Perform one step of the Schrödinger-Cable equation."""
    laplacian_term = - (D**2 / (2 * m)) * laplacian(psi, dx)
    dissipative_term = -1j * gamma * (psi - psi_rest)
    dpsi_dt = (-1j / D) * (laplacian_term + U_bio * psi) + dissipative_term
    return psi + dpsi_dt * dt

@jit
def solve_schrodinger_cable(psi_init, U_bio, x_grid, t_grid, D=1.0, gamma=0.1):
    """Solve Schrödinger-Cable PDE iteratively with Neumann BC."""
    psi_t_series = [psi_init]
    psi = psi_init
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    for t in t_grid[:-1]:
        psi = schrodinger_cable_step(psi, U_bio, dx, dt, D, gamma)
        psi_t_series.append(psi)

    return jnp.array(psi_t_series)

# ✅ 5. Define Optimal Control Algorithm (JAX-Based)
def cost_function(U_bio, psi_init, psi_target, x_grid, t_grid, lambda_reg=0.1):
    """Compute the cost function for optimal control."""
    psi_t_series = solve_schrodinger_cable(psi_init, U_bio, x_grid, t_grid)
    deviation_cost = jnp.sum(jnp.abs(psi_t_series - psi_target) ** 2)
    control_effort = lambda_reg * jnp.sum(jnp.abs(U_bio) ** 2)
    return deviation_cost + control_effort

grad_cost_function = jit(grad(cost_function))

def compute_optimal_control(psi_init, psi_target, x_grid, t_grid, learning_rate=0.01, num_iters=100):
    """Optimize bioelectric control field U_bio(x,t) using JAX-based optimization."""
    U_bio = jnp.zeros((len(x_grid), len(t_grid)))

    for i in range(num_iters):
        grad_U_bio = grad_cost_function(U_bio, psi_init, psi_target, x_grid, t_grid)
        U_bio = U_bio - learning_rate * grad_U_bio
        if i % 10 == 0:
            cost = cost_function(U_bio, psi_init, psi_target, x_grid, t_grid)
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return U_bio

# ✅ 6. Run Pipeline
file_path = "rstb20190765_si_001.xlsx"
groups = load_data(file_path)
spectrograms = spectral_analysis(groups)
cleaned_spectrograms = clean_spectrograms(spectrograms)

# Define space & time grid
Lx, Nx, T, Nt = 10.0, 100, 10.0, 1000
x_grid = jnp.linspace(0, Lx, Nx)
t_grid = jnp.linspace(0, T, Nt)

# Initialize bioelectric states
psi_init = jnp.exp(-((x_grid - 3.0) ** 2) / (2 * 0.5 ** 2))  # Cryptic worm state
psi_target = jnp.exp(-((x_grid - 5.0) ** 2) / (2 * 0.5 ** 2))  # Control worm state

# Compute optimal control field
U_bio_optimal = compute_optimal_control(psi_init, psi_target, x_grid, t_grid)

# Visualize optimal control field
plt.figure(figsize=(8, 5))
sns.heatmap(U_bio_optimal, cmap="coolwarm", cbar=True)
plt.xlabel("Time Step")
plt.ylabel("Spatial Position")
plt.title("Optimal Bioelectric Control Field U_bio(x,t) (JAX-Based)")
plt.show()

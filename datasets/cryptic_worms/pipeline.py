import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# -------------------------------------------------------------------------
# Mock "cryptic worm" vs. "control worm" data
# (In practice, replace these with your actual measurements.)
# -------------------------------------------------------------------------
# Suppose 'control_worm' is a 1D time series of length N_t.
N_t = 300
t_axis = np.linspace(0, 10, N_t)  # e.g. 0 to 10 seconds
control_worm = np.sin(2.0 * np.pi * 0.5 * t_axis)  # some synthetic wave

# We'll define the "cryptic worm" PDE on a spatial domain x in [0, L]
L = 5.0
N_x = 50
x_axis = np.linspace(0, L, N_x)

# PDE parameters (cable eqn)
D = 0.02  # diffusion
dt = 0.01  # time step
dx = x_axis[1] - x_axis[0]

# PDE simulation steps to match total time = 10s
steps = int(10.0 / dt)


# -------------------------------------------------------------------------
# PDE solver: 1D Cable equation + control
# dV/dt = D * d2V/dx2 - alpha * V + B(x)*u(x,t)
# For simplicity, let alpha=0 or some small coefficient. We'll do explicit Euler.
# -------------------------------------------------------------------------
def solve_cable_pde(V_init, control_2D):
    """
    V_init: shape (N_x,) initial voltage
    control_2D: shape (steps, N_x) or (N_x, steps).
                We'll assume shape = (steps, N_x). control_2D[t,i] = u(x_i, t)
    returns V_ts: shape (steps+1, N_x), PDE solution over time
    """
    V_ts = np.zeros((steps + 1, N_x))
    V_ts[0, :] = V_init.copy()

    # Zero-flux boundary conditions for simplicity: dV/dx=0 at x=0, x=L
    # We'll do an explicit Euler step:
    for n in range(steps):
        V_current = V_ts[n, :].copy()
        # second derivative in x using finite differences
        d2V_dx2 = np.zeros_like(V_current)
        for i in range(1, N_x - 1):
            d2V_dx2[i] = (
                V_current[i + 1] - 2.0 * V_current[i] + V_current[i - 1]
            ) / dx**2

        # simple eqn: dV/dt = D * d2V_dx2 + control
        # ignoring alpha * V for simplicity or alpha=0
        # control_2D[n,i] is the control at time-step n, space i
        dV_dt = D * d2V_dx2 + control_2D[n, :]

        V_next = V_current + dt * dV_dt

        # boundary: zero-flux => derivative=0 => replicate neighbor
        V_next[0] = V_next[1]
        V_next[-1] = V_next[-2]

        V_ts[n + 1, :] = V_next

    return V_ts


# -------------------------------------------------------------------------
# Wavelet analysis: we focus on final PDE state => we treat it as a time-series
# by sampling V at a certain spatial location or the average across space, etc.
# Then compare with "control_worm" wavelet spectrogram
# -------------------------------------------------------------------------
def fibonacci_scales(num=8):
    fib = [1, 1]
    for _ in range(2, num):
        fib.append(fib[-1] + fib[-2])
    fib = np.array(fib, dtype=float)
    return fib / fib[-1] * 20.0  # scale a bit


def wavelet_spectrum_1D(signal):
    scales = fibonacci_scales(10)
    cwtmatr, _ = pywt.cwt(signal, scales, "cmor1.0-1.0")
    return np.abs(cwtmatr)


spect_control = wavelet_spectrum_1D(control_worm)  # shape ~ (scales, time)


# -------------------------------------------------------------------------
# Cost function: PDE -> final PDE state -> extract a "representative" 1D signal
# e.g. the PDE solution at center x or the spatial average -> wavelet -> compare
# -------------------------------------------------------------------------
def cost_function_control(u_flat):
    """
    u_flat: flattened array of control shape=(steps*N_x,)
    We'll reshape it to (steps, N_x), solve PDE,
    then do wavelet comparison.
    """
    u_2D = u_flat.reshape((steps, N_x))

    # PDE init: cryptic worm state => let's do a localized bump
    V_init = np.exp(-((x_axis - 2.0) ** 2) / (2 * 0.5**2))

    V_ts = solve_cable_pde(V_init, u_2D)
    # We'll take the final PDE state and treat it as a time-series
    # by reading out V(t) at the center x= L/2 for t in [0..steps].
    # So we interpret: V_ts[t, i_mid], t=0..steps. Then wavelet-compare.
    i_mid = N_x // 2
    final_trace = V_ts[:, i_mid]  # shape (steps+1,)

    # We only have ~ steps=1000 in dt => that might not match control_worm length
    # Let's do a quick resample or the last portion to match length N_t.
    # We'll do a simple interpolation
    final_trace_times = np.linspace(0, 10, len(final_trace))
    worm_times = np.linspace(0, 10, N_t)
    final_trace_resampled = np.interp(worm_times, final_trace_times, final_trace)

    # wavelet
    cwt_cryptic = wavelet_spectrum_1D(final_trace_resampled)  # shape ~ (scales, N_t)
    # compare with cwt of control
    if cwt_cryptic.shape != spect_control.shape:
        # just in case minor dimension mismatch
        min_scale = min(cwt_cryptic.shape[0], spect_control.shape[0])
        min_time = min(cwt_cryptic.shape[1], spect_control.shape[1])
        cwt_cryptic = cwt_cryptic[:min_scale, :min_time]
        cwt_control = spect_control[:min_scale, :min_time]
    else:
        cwt_control = spect_control

    # sum of squared differences
    spectral_diff = np.sum((cwt_cryptic - cwt_control) ** 2)

    # regularize control
    reg_term = 0.001 * np.sum(u_flat**2)
    return spectral_diff + reg_term


# -------------------------------------------------------------------------
# Minimization: We'll do a naive approach with random initial control
# We'll use scipy.optimize.minimize to show proof-of-concept
# -------------------------------------------------------------------------
from scipy.optimize import Bounds


def main():
    # initial guess for control: zero
    init_u = np.zeros((steps, N_x), dtype=float)
    init_u_flat = init_u.ravel()

    # We can also impose some bounds on control if we want
    lb = -0.2 * np.ones_like(init_u_flat)
    ub = 0.2 * np.ones_like(init_u_flat)
    bounds = Bounds(lb, ub, keep_feasible=True)

    # run optimization
    result = minimize(
        cost_function_control,
        init_u_flat,
        method="L-BFGS-B",
        jac=None,  # using numerical gradient for simplicity
        bounds=bounds,
        options={"maxiter": 30, "disp": True},
    )

    print("Optimization done.")
    print("Final cost:", result.fun)
    U_opt_flat = result.x
    U_opt_2D = U_opt_flat.reshape((steps, N_x))

    # visualize final control
    plt.figure(figsize=(8, 4))
    sns.heatmap(U_opt_2D.T, cmap="coolwarm", cbar=True)
    plt.title("Optimal Control Field (time steps vs. space), cable PDE")
    plt.xlabel("Time step")
    plt.ylabel("Space index")
    plt.show()


if __name__ == "__main__":
    main()

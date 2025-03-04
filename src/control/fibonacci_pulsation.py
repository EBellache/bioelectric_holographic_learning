# fibonacci_pulsation.py
import jax.numpy as jnp


def fibonacci_sequence(n_terms):
    """Explicitly generate the first n_terms of the Fibonacci sequence."""
    seq = [1, 1]
    for _ in range(2, n_terms):
        seq.append(seq[-1] + seq[-2])
    return jnp.array(seq)


def fibonacci_pulsation(t_grid, amplitude=1.0):
    """Generate explicit Fibonacci-timed pulses for entropy reset."""
    fibonacci_intervals = fibonacci_sequence(len(t_grid))
    pulse_times = jnp.cumsum(fibonacci_intervals)
    pulses = jnp.zeros_like(t_grid)
    pulse_indices = pulse_times[pulse_times < t_grid[-1]].astype(int)

    pulses = pulses.at[pulse_indices].set(amplitude)
    return pulses

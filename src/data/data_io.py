"""
data_io.py

Defines functions for loading and saving bioelectric data,
e.g. cryptic worm signals, normal worm signals, etc.
"""

import jax.numpy as jnp
import pandas as pd


def load_worm_data_excel(file_path: str, sheet_name: str = "Sheet1"):
    """
    Load cryptic or normal worm signals from an Excel file.
    Returns a JAX array or dictionary of JAX arrays.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Example: filter columns, group by something, etc.
    # For demonstration, we'll just convert one column to jnp:
    yvar = df["yvar"].values
    return jnp.array(yvar)


def load_control_data():
    """
    Placeholder: loads 'control' bioelectric data for normal worms, etc.
    Possibly multiple columns or time series.
    """
    # In real code, you'd parse from CSV or Excel.
    # Return jnp arrays or a dict of them.
    pass


def save_results_to_csv(results_dict, file_path: str):
    """
    Saves final PDE solutions, optimal control, etc. to a CSV.
    """
    # demonstration:
    import csv

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # e.g. writer.writerow(["x", "u_final"])
        # for x, val in enumerate(results_dict["u_final"]):
        #    writer.writerow([x, float(val)])
    pass

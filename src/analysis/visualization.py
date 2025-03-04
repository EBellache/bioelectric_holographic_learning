# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_entropy(entropy_values, time_points, title="Entropy Over Time"):
    """Explicit plot of entropy dynamics."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, entropy_values, "-o")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_clusters(reduced_data, labels, title="Clustered Data (PCA)"):
    """Explicit scatter plot of clustered data."""
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(
        x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette=palette, s=80
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend(title="Clusters")
    plt.grid(True)
    plt.show()


def plot_spectral_decomposition(
    spectrogram, time_grid, frequencies, title="Spectral Decomposition"
):
    """Explicit visualization of spectral decomposition (Gabor spectrogram)."""
    plt.figure(figsize=(12, 6))
    plt.imshow(
        spectrogram,
        aspect="auto",
        cmap="inferno",
        extent=[time_grid[0], time_grid[-1], frequencies[-1], frequencies[0]],
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

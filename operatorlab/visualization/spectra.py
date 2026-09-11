"""Error spectrum visualization.

Computes 2D FFT of (pred - target) and plots error energy per wavenumber.
Shows where in frequency space the model fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def plot_error_spectrum(
    pred: Tensor,
    target: Tensor,
    title: str = "Error Spectrum",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot the radially-averaged error energy spectrum.

    Computes 2D FFT of (pred - target), then bins |error_hat|² by
    wavenumber magnitude to show error distribution across scales.

    Good models: low error at low wavenumbers, some drift at high wavenumbers.
    Bad models: error accumulates at specific frequency bands (instability).

    Args:
        pred: Predicted field, shape (batch, H, W, C) or (H, W).
        target: Target field, same shape.
        title: Plot title.
        save_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    # Handle various input shapes
    if pred.ndim == 4:
        pred = pred[0, ..., 0]
    elif pred.ndim == 3:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0, ..., 0]
    elif target.ndim == 3:
        target = target[0]

    error = (pred - target).detach().cpu().numpy()
    h, w = error.shape

    # 2D FFT of error
    error_hat = np.fft.fft2(error)
    error_energy = np.abs(error_hat) ** 2

    # Also compute target spectrum for comparison
    target_np = target.detach().cpu().numpy()
    target_hat = np.fft.fft2(target_np)
    target_energy = np.abs(target_hat) ** 2

    # Radially average
    kx = np.fft.fftfreq(h, d=1.0 / h)
    ky = np.fft.fftfreq(w, d=1.0 / w)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_max = int(min(h, w) // 2)
    k_bins = np.arange(0.5, k_max + 0.5, 1.0)

    error_radial = np.zeros(len(k_bins))
    target_radial = np.zeros(len(k_bins))

    for i, k_val in enumerate(k_bins):
        mask = (k_mag >= k_val - 0.5) & (k_mag < k_val + 0.5)
        if mask.any():
            error_radial[i] = error_energy[mask].mean()
            target_radial[i] = target_energy[mask].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_bins, target_radial, "b-", label="Target spectrum", linewidth=1.5)
    ax.semilogy(k_bins, error_radial, "r-", label="Error spectrum", linewidth=1.5)
    ax.fill_between(k_bins, error_radial, alpha=0.2, color="red")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

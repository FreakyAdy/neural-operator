"""Field visualization: plots of input, prediction, target, and error fields."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_field_comparison(
    a: Tensor,
    pred: Tensor,
    target: Tensor,
    sample_idx: int = 0,
    title: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot side-by-side comparison: input, prediction, target, error.

    Args:
        a: Input field, shape (batch, H, W, C).
        pred: Predicted field, shape (batch, H, W, C).
        target: Target field, shape (batch, H, W, C).
        sample_idx: Which sample in the batch to plot.
        title: Plot title.
        save_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    a_np = a[sample_idx, ..., 0].detach().cpu().numpy()
    pred_np = pred[sample_idx, ..., 0].detach().cpu().numpy()
    target_np = target[sample_idx, ..., 0].detach().cpu().numpy()
    error_np = pred_np - target_np

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im0 = axes[0].imshow(a_np, cmap="RdBu_r", aspect="equal")
    axes[0].set_title("Input a(x)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_np, cmap="RdBu_r", aspect="equal")
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(target_np, cmap="RdBu_r", aspect="equal")
    axes[2].set_title("Ground Truth")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(error_np, cmap="coolwarm", aspect="equal")
    axes[3].set_title("Error (pred - target)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_vorticity_field(
    omega: Tensor,
    title: str = "Vorticity",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a single vorticity field with streamlines.

    Args:
        omega: Vorticity field, shape (H, W) or (1, H, W, 1).
        title: Plot title.
        save_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    if omega.ndim == 4:
        omega = omega[0, ..., 0]
    omega_np = omega.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(omega_np, cmap="RdBu_r", aspect="equal")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

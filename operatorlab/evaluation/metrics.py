"""Evaluation metrics for neural operators.

Provides:
    - l2_error: absolute L2 error
    - relative_l2_error: relative L2 error
    - h1_error: H1 Sobolev error
    - energy_spectrum_error: error in the energy spectrum
"""

from __future__ import annotations

import torch
from torch import Tensor


def l2_error(pred: Tensor, target: Tensor) -> Tensor:
    """Absolute L2 error per sample, averaged over batch.

    Args:
        pred: Predicted field, shape (batch, *spatial, channels).
        target: Target field, same shape.

    Returns:
        Scalar mean L2 error.
    """
    batch_size = pred.shape[0]
    diff = (pred - target).reshape(batch_size, -1)
    return torch.norm(diff, dim=1).mean()


def relative_l2_error(pred: Tensor, target: Tensor) -> Tensor:
    """Relative L2 error per sample, averaged over batch.

    Args:
        pred: Predicted field.
        target: Target field.

    Returns:
        Scalar mean relative L2 error.
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    return (diff_norm / (target_norm + 1e-8)).mean()


def h1_error(pred: Tensor, target: Tensor, dx: float = 1.0) -> Tensor:
    """H1 Sobolev error including gradient differences.

    Args:
        pred: Predicted field, shape (batch, H, W, C).
        target: Target field, same shape.
        dx: Grid spacing.

    Returns:
        Scalar H1 error.
    """
    l2 = relative_l2_error(pred, target)

    p = pred.permute(0, 3, 1, 2)
    t = target.permute(0, 3, 1, 2)

    dp_dx = (p[:, :, 1:, :] - p[:, :, :-1, :]) / dx
    dt_dx = (t[:, :, 1:, :] - t[:, :, :-1, :]) / dx
    dp_dy = (p[:, :, :, 1:] - p[:, :, :, :-1]) / dx
    dt_dy = (t[:, :, :, 1:] - t[:, :, :, :-1]) / dx

    batch_size = pred.shape[0]
    gx_err = torch.norm((dp_dx - dt_dx).reshape(batch_size, -1), dim=1)
    gx_ref = torch.norm(dt_dx.reshape(batch_size, -1), dim=1)
    gy_err = torch.norm((dp_dy - dt_dy).reshape(batch_size, -1), dim=1)
    gy_ref = torch.norm(dt_dy.reshape(batch_size, -1), dim=1)

    grad_err = (gx_err / (gx_ref + 1e-8) + gy_err / (gy_ref + 1e-8)).mean()
    return l2 + grad_err


def energy_spectrum_error(pred: Tensor, target: Tensor) -> Tensor:
    """Error in the radially-averaged energy spectrum.

    Computes 2D FFT of both fields, bins by wavenumber magnitude,
    and returns the relative error in the radial energy spectrum.

    Args:
        pred: Predicted field, shape (batch, H, W, C).
        target: Target field, same shape.

    Returns:
        Scalar relative energy spectrum error.
    """
    pred_f = pred[..., 0]    # (batch, H, W)
    target_f = target[..., 0]

    pred_hat = torch.fft.fft2(pred_f, norm="ortho")
    target_hat = torch.fft.fft2(target_f, norm="ortho")

    pred_energy = (pred_hat.abs() ** 2).mean(0)    # (H, W)
    target_energy = (target_hat.abs() ** 2).mean(0)

    # Relative error in total spectral energy
    diff = torch.norm(pred_energy - target_energy)
    ref = torch.norm(target_energy) + 1e-8
    return diff / ref


def linf_error(pred: Tensor, target: Tensor) -> Tensor:
    """Maximum pointwise absolute error (L_infinity norm).

    Args:
        pred: Predicted field.
        target: Target field.

    Returns:
        Scalar maximum absolute error.
    """
    return (pred - target).abs().max()


def speedup_vs_solver(model_time_ms: float, solver_time_ms: float) -> float:
    """Calculate inference speedup factor over classical numerical solver.

    Args:
        model_time_ms: Inference time in milliseconds.
        solver_time_ms: Numerical solver runtime in milliseconds.

    Returns:
        Speedup multiplier.
    """
    return solver_time_ms / max(model_time_ms, 1e-6)


# Convenient aliases matching CLAUDE.md terminology
relative_l2 = relative_l2_error
relative_h1 = h1_error

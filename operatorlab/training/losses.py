"""Loss functions for neural operator training.

Provides:
    - relative_l2: primary evaluation metric
    - h1_loss: Sobolev H1 loss penalizing rough solutions
    - pde_residual_loss: physics-informed regularization
    - OperatorLoss: combined weighted loss module
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from operatorlab.physics.base import PDEProblem


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Relative L2 error between predicted and target fields.

    Computes ||pred - target||_2 / ||target||_2 per sample,
    then averages over the batch.

    Args:
        pred: Predicted field, shape (batch, *spatial, channels).
        target: Target field, shape (batch, *spatial, channels).

    Returns:
        Scalar tensor with mean relative L2 error.
    """
    batch_size = pred.shape[0]
    # Flatten spatial and channel dims: (batch, -1)
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    diff_norm = torch.norm(pred_flat - target_flat, dim=1)  # (batch,)
    target_norm = torch.norm(target_flat, dim=1)  # (batch,)

    rel_err = diff_norm / (target_norm + 1e-8)
    return rel_err.mean()


def h1_loss(pred: Tensor, target: Tensor, dx: float = 1.0) -> Tensor:
    """H1 (Sobolev) loss: L2 error on both values and gradients.

    ||pred - target||_{H^1} = ||pred - target||_{L^2}
                              + ||∇pred - ∇target||_{L^2}

    Uses finite-difference approximation for spatial gradients.
    Expects input shape (batch, H, W, channels).

    Args:
        pred: Predicted field, shape (batch, H, W, channels).
        target: Target field, shape (batch, H, W, channels).
        dx: Grid spacing for finite-difference gradient computation.

    Returns:
        Scalar tensor with H1 loss.
    """
    # L2 part
    l2 = relative_l2(pred, target)

    # Permute to (batch, channels, H, W) for spatial gradient computation
    p = pred.permute(0, 3, 1, 2)   # (batch, C, H, W)
    t = target.permute(0, 3, 1, 2)  # (batch, C, H, W)

    # Finite-difference gradients along H and W dimensions
    # df/dx ≈ (f[..., i+1, :] - f[..., i, :]) / dx
    dp_dx = (p[:, :, 1:, :] - p[:, :, :-1, :]) / dx  # (batch, C, H-1, W)
    dt_dx = (t[:, :, 1:, :] - t[:, :, :-1, :]) / dx

    dp_dy = (p[:, :, :, 1:] - p[:, :, :, :-1]) / dx  # (batch, C, H, W-1)
    dt_dy = (t[:, :, :, 1:] - t[:, :, :, :-1]) / dx

    # Relative gradient error
    batch_size = pred.shape[0]

    grad_x_diff = (dp_dx - dt_dx).reshape(batch_size, -1)
    grad_x_ref = dt_dx.reshape(batch_size, -1)
    grad_x_err = torch.norm(grad_x_diff, dim=1) / (torch.norm(grad_x_ref, dim=1) + 1e-8)

    grad_y_diff = (dp_dy - dt_dy).reshape(batch_size, -1)
    grad_y_ref = dt_dy.reshape(batch_size, -1)
    grad_y_err = torch.norm(grad_y_diff, dim=1) / (torch.norm(grad_y_ref, dim=1) + 1e-8)

    grad_err = (grad_x_err + grad_y_err).mean()

    return l2 + grad_err


def pde_residual_loss(
    pred: Tensor,
    pde: PDEProblem,
    t: float,
    weight: float = 0.1,
) -> Tensor:
    """Physics-informed regularization term.

    Computes the PDE residual of the predicted field and returns
    a weighted scalar loss. Add to data loss:
        total_loss = data_loss + weight * pde_loss

    Args:
        pred: Predicted field, shape (batch, *spatial, channels).
        pde: PDE problem instance providing compute_residual().
        t: Time at which to evaluate the PDE residual.
        weight: Scaling factor for the residual loss.

    Returns:
        Scalar tensor with weighted PDE residual loss.
    """
    residual = pde.compute_residual(pred, t)  # (batch, *spatial, ...)
    batch_size = pred.shape[0]
    residual_flat = residual.reshape(batch_size, -1)
    residual_norm = torch.norm(residual_flat, dim=1).mean()
    return weight * residual_norm


class OperatorLoss(nn.Module):
    """Combined loss for neural operator training.

    Supports weighted combination of:
        - Relative L2 loss (data fidelity)
        - H1 Sobolev loss (gradient regularization)
        - PDE residual loss (physics-informed regularization)

    Args:
        l2_weight: Weight for relative L2 loss.
        h1_weight: Weight for H1 loss.
        pde_weight: Weight for PDE residual loss.
        pde: Optional PDE problem instance (required if pde_weight > 0).
        dx: Grid spacing for H1 loss computation.
        t: Time for PDE residual evaluation.
    """

    def __init__(
        self,
        l2_weight: float = 1.0,
        h1_weight: float = 0.0,
        pde_weight: float = 0.0,
        pde: Optional[PDEProblem] = None,
        dx: float = 1.0,
        t: float = 1.0,
    ) -> None:
        super().__init__()
        self.l2_weight = l2_weight
        self.h1_weight = h1_weight
        self.pde_weight = pde_weight
        self.pde = pde
        self.dx = dx
        self.t = t

        if pde_weight > 0 and pde is None:
            raise ValueError("pde must be provided when pde_weight > 0")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute combined loss.

        Args:
            pred: Predicted field, shape (batch, *spatial, channels).
            target: Target field, shape (batch, *spatial, channels).

        Returns:
            Scalar combined loss tensor.
        """
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if self.l2_weight > 0:
            loss = loss + self.l2_weight * relative_l2(pred, target)

        if self.h1_weight > 0:
            loss = loss + self.h1_weight * h1_loss(pred, target, self.dx)

        if self.pde_weight > 0 and self.pde is not None:
            loss = loss + pde_residual_loss(
                pred, self.pde, self.t, weight=self.pde_weight,
            )

        return loss

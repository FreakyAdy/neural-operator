"""Geometry OOD stress testing.

Evaluates how neural operators respond when spatial domain geometry shifts
from regular Cartesian grids to irregular, non-equispaced, or warped coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.evaluation.ood import warp_grid_coordinates
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class GeometryStressResult:
    """Outcome of geometry shift stress test."""

    distortion_strength: float
    base_l2: float
    warped_l2: float
    degradation_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def evaluate_geometry_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    resolution: int = 64,
    distortion_strength: float = 0.08,
    n_samples: int = 20,
    device: str = "cpu",
) -> GeometryStressResult:
    """Stress test operator on warped / irregular domain geometry."""
    model.eval()

    base_data = pde.generate_dataset(n_samples=n_samples, resolution=resolution, seed=301)
    a = base_data["a"].to(device)
    u = base_data["u"].to(device)

    with torch.no_grad():
        pred_base = model(a)
    base_l2 = relative_l2_error(pred_base, u).item()

    # Generate warped coordinates
    warped_grid = warp_grid_coordinates(resolution, distortion_strength=distortion_strength, device=device)
    warped_grid_expanded = warped_grid.expand(len(a), -1, -1, -1)

    with torch.no_grad():
        pred_warped = model(a, grid=warped_grid_expanded)
    warped_l2 = relative_l2_error(pred_warped, u).item()

    degradation = warped_l2 / max(base_l2, 1e-8)

    if degradation < 1.4:
        status = "PASS"
    elif degradation < 2.5:
        status = "WARN"
    else:
        status = "FAIL"

    return GeometryStressResult(
        distortion_strength=distortion_strength,
        base_l2=base_l2,
        warped_l2=warped_l2,
        degradation_ratio=degradation,
        status=status,
        message=f"Square → irregular domain ({degradation:.2f}x)",
    )

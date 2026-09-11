"""Boundary OOD stress testing.

Evaluates how neural operators handle boundary condition shifts
(e.g. Periodic domain models evaluated on non-periodic boundaries or boundary dampening).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class BoundaryStressResult:
    """Outcome of boundary condition stress test."""

    boundary_condition: str
    base_l2: float
    boundary_l2: float
    degradation_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def apply_boundary_damping(a: Tensor, width: int = 3, factor: float = 0.0) -> Tensor:
    """Enforce Dirichlet zero or damped boundary condition along perimeter edges."""
    a_damped = a.clone()
    a_damped[:, :width, :, :] *= factor
    a_damped[:, -width:, :, :] *= factor
    a_damped[:, :, :width, :] *= factor
    a_damped[:, :, -width:, :] *= factor
    return a_damped


def evaluate_boundary_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    resolution: int = 64,
    boundary_width: int = 3,
    n_samples: int = 20,
    device: str = "cpu",
) -> BoundaryStressResult:
    """Stress test operator when boundary conditions shift from periodic to Dirichlet edge damping."""
    model.eval()

    base_data = pde.generate_dataset(n_samples=n_samples, resolution=resolution, seed=401)
    a = base_data["a"].to(device)
    u = base_data["u"].to(device)

    with torch.no_grad():
        pred_base = model(a)
    base_l2 = relative_l2_error(pred_base, u).item()

    # Apply Dirichlet boundary damping on input and ground truth
    a_dirichlet = apply_boundary_damping(a, width=boundary_width, factor=0.0)
    u_dirichlet = apply_boundary_damping(u, width=boundary_width, factor=0.0)

    with torch.no_grad():
        pred_dirichlet = model(a_dirichlet)
    dirichlet_l2 = relative_l2_error(pred_dirichlet, u_dirichlet).item()

    degradation = dirichlet_l2 / max(base_l2, 1e-8)

    if degradation < 1.5:
        status = "PASS"
    elif degradation < 3.0:
        status = "WARN"
    else:
        status = "FAIL"

    return BoundaryStressResult(
        boundary_condition="Periodic → Dirichlet",
        base_l2=base_l2,
        boundary_l2=dirichlet_l2,
        degradation_ratio=degradation,
        status=status,
        message=f"Periodic → Dirichlet ({degradation:.2f}x)",
    )

"""Resolution OOD stress testing.

Evaluates neural operator discretization invariance across unseen higher resolutions
(e.g. 64x64 trained -> 128x128, 256x256, 512x512).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class ResolutionStressResult:
    """Outcome of resolution stress test."""

    base_res: int
    target_res: int
    base_l2: float
    target_l2: float
    degradation_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def evaluate_resolution_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    base_res: int = 64,
    target_res: int = 256,
    n_samples: int = 20,
    device: str = "cpu",
) -> ResolutionStressResult:
    """Stress-test zero-shot resolution transfer."""
    model.eval()

    # Generate baseline data
    base_data = pde.generate_dataset(n_samples=n_samples, resolution=base_res, seed=101)
    a_base = base_data["a"].to(device)
    u_base = base_data["u"].to(device)

    with torch.no_grad():
        pred_base = model(a_base)
    base_l2 = relative_l2_error(pred_base, u_base).item()

    # Generate target resolution data
    target_data = pde.generate_dataset(n_samples=n_samples, resolution=target_res, seed=102)
    a_target = target_data["a"].to(device)
    u_target = target_data["u"].to(device)

    with torch.no_grad():
        pred_target = model(a_target)
    target_l2 = relative_l2_error(pred_target, u_target).item()

    degradation = target_l2 / max(base_l2, 1e-8)

    if degradation < 1.5:
        status = "PASS"
        msg = f"{base_res} → {target_res} preserved ({degradation:.2f}x drift)"
    elif degradation < 3.0:
        status = "WARN"
        msg = f"{base_res} → {target_res} moderate drift ({degradation:.2f}x)"
    else:
        status = "FAIL"
        msg = f"{base_res} → {target_res} severe degradation ({degradation:.2f}x)"

    return ResolutionStressResult(
        base_res=base_res,
        target_res=target_res,
        base_l2=base_l2,
        target_l2=target_l2,
        degradation_ratio=degradation,
        status=status,
        message=msg,
    )

"""Long-horizon autoregressive rollout stress testing.

Evaluates how well a neural operator sustains multi-step temporal rollouts (t = 1 -> 20+)
without numerical blowup, unphysical dissipation, or error cascading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class LongHorizonStressResult:
    """Outcome of long-horizon rollout stress test."""

    target_steps: int
    stable_steps: int
    final_norm_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def evaluate_long_horizon_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    resolution: int = 64,
    target_steps: int = 20,
    device: str = "cpu",
) -> LongHorizonStressResult:
    """Stress test autoregressive rollout stability for t = 1 -> target_steps."""
    model.eval()

    # Generate a single trajectory initial condition
    data = pde.generate_dataset(n_samples=1, resolution=resolution, seed=601)
    curr = data["a"].to(device)
    init_norm = curr.norm().item()

    grid = None
    if hasattr(model, "make_grid"):
        grid = model.make_grid((resolution, resolution), device).expand(1, -1, -1, -1)

    stable_steps = 0
    final_norm_ratio = 1.0

    with torch.no_grad():
        for step in range(1, target_steps + 1):
            curr = model(curr, grid=grid)
            curr_norm = curr.norm().item()

            if torch.isnan(curr).any() or torch.isinf(curr).any():
                final_norm_ratio = float("nan")
                break

            ratio = curr_norm / max(init_norm, 1e-6)
            if ratio > 10.0 or ratio < 0.01:
                # Numerical blowup or complete unphysical dampening
                final_norm_ratio = ratio
                break

            stable_steps = step
            final_norm_ratio = ratio

    if stable_steps >= target_steps:
        status = "PASS"
    elif stable_steps >= target_steps // 2:
        status = "WARN"
    else:
        status = "FAIL"

    return LongHorizonStressResult(
        target_steps=target_steps,
        stable_steps=stable_steps,
        final_norm_ratio=final_norm_ratio,
        status=status,
        message=f"t = 1 → {target_steps} ({stable_steps}/{target_steps} stable steps)",
    )

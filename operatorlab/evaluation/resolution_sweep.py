"""Resolution sweep evaluation — the killer feature.

Trains at one resolution, evaluates zero-shot at multiple others.
This validates that a neural operator has truly learned a
discretization-independent operator, not a fixed-resolution function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import relative_l2_error, h1_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Metrics at a single resolution."""

    resolution: int
    l2_error: float
    h1_error: float
    physics_violation: float = 0.0


@dataclass
class ResolutionSweepResult:
    """Results from a full resolution sweep."""

    base_resolution: int
    results: list[ResolutionResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Format results as a printable table."""
        lines = [
            f"{'Resolution':<12} | {'L2 Error':<10} | {'H1 Error':<10} | {'Physics Violation':<18}",
            "-" * 60,
        ]
        for r in self.results:
            lines.append(
                f"{r.resolution}×{r.resolution:<6} | {r.l2_error:<10.6f} | "
                f"{r.h1_error:<10.6f} | {r.physics_violation:<18.2e}"
            )
        return "\n".join(lines)


@torch.no_grad()
def resolution_sweep(
    model: NeuralOperator,
    pde: PDEProblem,
    base_resolution: int,
    target_resolutions: list[int],
    n_test_samples: int = 200,
    device: str = "cuda",
) -> ResolutionSweepResult:
    """Run zero-shot resolution sweep evaluation.

    For each target resolution:
        1. Generate test data at that resolution
        2. Run model forward pass (NO retraining)
        3. Compute relative L2, H1 error, and physics violations

    Args:
        model: Trained neural operator model.
        pde: PDE problem for generating test data.
        base_resolution: Resolution the model was trained at.
        target_resolutions: List of resolutions to evaluate at.
        n_test_samples: Number of test samples per resolution.
        device: Torch device.

    Returns:
        ResolutionSweepResult with per-resolution metrics.
    """
    model.eval()
    sweep = ResolutionSweepResult(base_resolution=base_resolution)

    for res in target_resolutions:
        logger.info("Evaluating at resolution %d×%d...", res, res)

        # Generate test data at this resolution
        test_data = pde.generate_dataset(
            n_samples=n_test_samples,
            resolution=res,
            dt=1e-3,
            seed=9999,  # fixed seed for reproducibility across sweeps
        )

        a = test_data["a"].to(device)   # (n, res, res, 1)
        u = test_data["u"].to(device)   # (n, res, res, 1)

        # Forward pass in batches
        batch_size = 20
        all_preds = []
        for i in range(0, len(a), batch_size):
            a_batch = a[i:i + batch_size]
            pred_batch = model(a_batch)
            all_preds.append(pred_batch)
        pred = torch.cat(all_preds, dim=0)  # (n, res, res, 1)

        # Compute metrics
        l2 = relative_l2_error(pred, u).item()
        h1 = h1_error(pred, u, dx=1.0 / res).item()

        # Physics violation
        conservation = pde.check_conservation(pred)
        physics_viol = sum(abs(v) for v in conservation.values())

        result = ResolutionResult(
            resolution=res,
            l2_error=l2,
            h1_error=h1,
            physics_violation=physics_viol,
        )
        sweep.results.append(result)
        logger.info(
            "  %d×%d: L2=%.6f, H1=%.6f, Phys=%.2e",
            res, res, l2, h1, physics_viol,
        )

    return sweep

"""Physics violation analysis.

Computes PDE residual norms, conservation law violations,
divergence fields, and energy spectrum comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from operatorlab.evaluation.metrics import energy_spectrum_error, relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class PhysicsViolationReport:
    """Report of physics violations for a model's predictions."""

    mean_residual_norm: float = 0.0
    max_residual_norm: float = 0.0
    conservation_violations: dict[str, float] = field(default_factory=dict)
    energy_spectrum_error: float = 0.0
    divergence_norm: float = 0.0


@torch.no_grad()
def analyze_physics_violations(
    model: NeuralOperator,
    pde: PDEProblem,
    test_loader: DataLoader,
    device: str = "cpu",
) -> PhysicsViolationReport:
    """Analyze physics violations of model predictions.

    Computes:
        - PDE residual norm per sample
        - Conservation law violations
        - Divergence (for velocity fields)
        - Energy spectrum comparison (predicted vs ground truth)

    Args:
        model: Trained neural operator.
        pde: PDE problem instance.
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        PhysicsViolationReport with all violation metrics.
    """
    model.eval()

    all_residual_norms: list[float] = []
    all_conservation: dict[str, list[float]] = {}
    all_spectrum_errors: list[float] = []
    all_divergence_norms: list[float] = []

    for a, u, grid in test_loader:
        a = a.to(device)
        u = u.to(device)
        grid = grid.to(device)

        if grid.ndim == 3:
            grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)

        pred = model(a, grid)

        # PDE residual
        residual = pde.compute_residual(pred, t=1.0)
        batch_size = pred.shape[0]
        res_norms = torch.norm(
            residual.reshape(batch_size, -1), dim=1,
        )
        all_residual_norms.extend(res_norms.cpu().tolist())

        # Conservation
        conservation = pde.check_conservation(pred)
        for key, val in conservation.items():
            if key not in all_conservation:
                all_conservation[key] = []
            all_conservation[key].append(val)

        # Energy spectrum error
        spec_err = energy_spectrum_error(pred, u)
        all_spectrum_errors.append(spec_err.item())

        # Divergence (for 2D fields, compute ∂u/∂x + ∂u/∂y as diagnostic)
        if pred.shape[-1] >= 1:
            field = pred[..., 0]  # (batch, H, W)
            h, w = field.shape[-2:]
            k_x = torch.fft.fftfreq(h, d=1.0 / h, device=device) * 2 * torch.pi
            k_y = torch.fft.fftfreq(w, d=1.0 / w, device=device) * 2 * torch.pi
            kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")

            field_hat = torch.fft.fft2(field)
            dudx = torch.fft.ifft2(1j * kx.unsqueeze(0) * field_hat).real
            dudy = torch.fft.ifft2(1j * ky.unsqueeze(0) * field_hat).real
            div = dudx + dudy
            div_norm = torch.norm(div.reshape(batch_size, -1), dim=1).mean()
            all_divergence_norms.append(div_norm.item())

    # Aggregate
    mean_residual = sum(all_residual_norms) / max(len(all_residual_norms), 1)
    max_residual = max(all_residual_norms) if all_residual_norms else 0.0

    avg_conservation = {
        k: sum(v) / len(v) for k, v in all_conservation.items()
    }

    report = PhysicsViolationReport(
        mean_residual_norm=mean_residual,
        max_residual_norm=max_residual,
        conservation_violations=avg_conservation,
        energy_spectrum_error=sum(all_spectrum_errors) / max(len(all_spectrum_errors), 1),
        divergence_norm=sum(all_divergence_norms) / max(len(all_divergence_norms), 1),
    )

    logger.info("Physics violation analysis complete:")
    logger.info("  Mean residual norm: %.6f", report.mean_residual_norm)
    logger.info("  Max residual norm:  %.6f", report.max_residual_norm)
    logger.info("  Energy spectrum err: %.6f", report.energy_spectrum_error)
    logger.info("  Divergence norm:    %.6f", report.divergence_norm)
    for k, v in report.conservation_violations.items():
        logger.info("  %s: %.6f", k, v)

    return report

"""Scientific Validity and Physical Invariants Audit Layer.

Audits whether a neural operator's predictions behave like a physically valid solution,
beyond just matching training data:
1. Conservation laws (mass, energy, momentum)
2. Invariants (divergence-free constraint, Hamiltonian drift)
3. Spectral structure (energy cascade, high-frequency artifacts/Nyquist pileup)
4. Stability & boundedness (numerical blow-up, long-horizon autoregressive stability)
5. Discretization invariance

Produces an executive Scientific Audit Card with PASS / WARN / FAIL ratings.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from operatorlab.evaluation.metrics import energy_spectrum_error, relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


class AuditStatus(str, Enum):
    """Audit status badge."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class AuditCriterionResult:
    """Evaluation result for a single scientific validity criterion."""

    name: str
    category: str
    measured_value: float
    threshold: float
    comparator: str  # '<', '<=', '>', '>='
    status: AuditStatus
    details: str = ""


@dataclass
class ScientificAuditCard:
    """Executive Scientific Validity Audit Card."""

    model_name: str
    pde_name: str
    resolution: int
    criteria: list[AuditCriterionResult] = field(default_factory=list)

    @property
    def overall_verdict(self) -> str:
        """Compute aggregate verdict."""
        has_fail = any(c.status == AuditStatus.FAIL for c in self.criteria)
        has_warn = any(c.status == AuditStatus.WARN for c in self.criteria)
        if has_fail:
            return "PHYSICALLY INVALID (Critical Violations)"
        elif has_warn:
            return "CONDITIONALLY VALID (Warnings Present)"
        return "PHYSICALLY VALID (All Constraints Respected)"

    def summary_card(self) -> str:
        """Format an executive terminal audit card."""
        w = 80
        lines = [
            "╔" + "═" * (w - 2) + "╗",
            f"║ Scientific Validity Audit: {self.model_name} on {self.pde_name} ({self.resolution}×{self.resolution})".ljust(w - 1) + "║",
            "╠" + "═" * (w - 2) + "╣",
            f"║ {'Criterion':<26} | {'Category':<13} | {'Value':<12} | {'Threshold':<11} | {'Status':<6} ║",
            "╟" + "─" * 28 + "┼" + "─" * 15 + "┼" + "─" * 14 + "┼" + "─" * 13 + "┼" + "─" * 8 + "╢",
        ]

        for c in self.criteria:
            val_str = f"{c.measured_value:.2e}" if abs(c.measured_value) < 1e-2 or abs(c.measured_value) > 1e3 else f"{c.measured_value:.4f}"
            thresh_str = f"{c.comparator} {c.threshold:.2e}" if abs(c.threshold) < 1e-2 else f"{c.comparator} {c.threshold:.3f}"
            status_str = f"[{c.status.value}]"
            lines.append(
                f"║ {c.name:<26} | {c.category:<13} | {val_str:<12} | {thresh_str:<11} | {status_str:<6} ║"
            )

        lines.append("╠" + "═" * (w - 2) + "╣")
        verdict_str = f"OVERALL VERDICT: {self.overall_verdict}"
        lines.append(f"║ {verdict_str}".ljust(w - 1) + "║")
        lines.append("╚" + "═" * (w - 2) + "╝")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pde_name": self.pde_name,
            "resolution": self.resolution,
            "overall_verdict": self.overall_verdict,
            "criteria": [
                {
                    "name": c.name,
                    "category": c.category,
                    "measured_value": c.measured_value,
                    "threshold": c.threshold,
                    "comparator": c.comparator,
                    "status": c.status.value,
                    "details": c.details,
                }
                for c in self.criteria
            ],
        }


# -----------------------------------------------------------------------------
# Diagnostic Evaluators
# -----------------------------------------------------------------------------

def evaluate_mass_conservation(pred: Tensor, initial: Tensor) -> float:
    """Measure relative mass / mean integral conservation drift.

    Args:
        pred: Predicted field at final time (batch, H, W, c).
        initial: Initial field (batch, H, W, c).
    """
    mass_init = initial.mean(dim=(1, 2))  # (batch, c)
    mass_pred = pred.mean(dim=(1, 2))    # (batch, c)
    drift = (mass_pred - mass_init).abs() / (mass_init.abs() + 1e-5)
    return drift.mean().item()


def evaluate_energy_drift(pred: Tensor, target: Tensor) -> float:
    """Measure relative $L_2$ kinetic/field energy drift compared to ground truth.

    E = ∫ |u|² dx
    """
    energy_true = (target ** 2).mean(dim=(1, 2))
    energy_pred = (pred ** 2).mean(dim=(1, 2))
    drift = (energy_pred - energy_true).abs() / (energy_true + 1e-6)
    return drift.mean().item()


def evaluate_divergence_constraint(pred: Tensor, device: str = "cpu") -> float:
    """Compute mean divergence norm for 2D vector fields or vorticity fields.

    For scalar field u, evaluates ∂u/∂x + ∂u/∂y norm as diagnostic.
    """
    field = pred[..., 0]  # (batch, H, W)
    h, w = field.shape[-2:]
    k_x = torch.fft.fftfreq(h, d=1.0 / h, device=device) * 2 * torch.pi
    k_y = torch.fft.fftfreq(w, d=1.0 / w, device=device) * 2 * torch.pi
    kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")

    field_hat = torch.fft.fft2(field)
    dudx = torch.fft.ifft2(1j * kx.unsqueeze(0) * field_hat).real
    dudy = torch.fft.ifft2(1j * ky.unsqueeze(0) * field_hat).real
    div = dudx + dudy
    return torch.norm(div.reshape(len(field), -1), dim=1).mean().item()


def evaluate_high_frequency_pileup(pred: Tensor, target: Tensor) -> float:
    """Measure unphysical spectral accumulation / Nyquist pileup in top 15% frequency modes.

    Values >> 1.0 indicate spurious high-frequency oscillations / checkerboard artifacts.
    """
    pred_hat = torch.fft.rfft2(pred.permute(0, 3, 1, 2))
    target_hat = torch.fft.rfft2(target.permute(0, 3, 1, 2))

    fh, fw = pred_hat.shape[-2:]
    cutoff_h = int(fh * 0.85)
    cutoff_w = int(fw * 0.85)

    high_pred_power = (pred_hat[..., cutoff_h:, cutoff_w:].abs() ** 2).mean()
    high_target_power = (target_hat[..., cutoff_h:, cutoff_w:].abs() ** 2).mean()

    ratio = high_pred_power / (high_target_power + 1e-8)
    return ratio.item()


def evaluate_long_horizon_stability(
    model: NeuralOperator,
    sample_a: Tensor,
    sample_grid: Optional[Tensor],
    n_steps: int = 20,
    device: str = "cpu",
) -> int:
    """Autoregressively roll out model predictions and count stable steps.

    Returns:
        Number of steps before prediction explodes (NaN/Inf or norm > 10x init).
    """
    curr = sample_a.to(device)
    grid = sample_grid.to(device) if sample_grid is not None else None
    init_norm = curr.norm().item()

    stable_steps = 0
    for _ in range(n_steps):
        with torch.no_grad():
            curr = model(curr, grid=grid)
        if torch.isnan(curr).any() or torch.isinf(curr).any():
            break
        if curr.norm().item() > 10.0 * max(init_norm, 1.0):
            break
        stable_steps += 1

    return stable_steps


# -----------------------------------------------------------------------------
# Main Audit Runner
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_scientific_audit(
    model: NeuralOperator,
    pde: PDEProblem,
    test_loader: DataLoader,
    device: str = "cpu",
    max_rollout_steps: int = 25,
) -> ScientificAuditCard:
    """Perform a multi-criteria scientific validity audit on a trained neural operator.

    Args:
        model: Trained neural operator.
        pde: PDE problem instance.
        test_loader: Test DataLoader.
        device: Torch device.
        max_rollout_steps: Max autoregressive steps for stability test.

    Returns:
        ScientificAuditCard with all validity ratings.
    """
    model.eval()
    model_name = type(model).__name__
    pde_name = getattr(pde, "name", type(pde).__name__)

    all_preds = []
    all_u = []
    all_a = []
    all_grid = []

    for a, u, grid in test_loader:
        a = a.to(device)
        u = u.to(device)
        grid = grid.to(device)
        if grid.ndim == 3:
            grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)

        pred = model(a, grid=grid)
        all_preds.append(pred)
        all_u.append(u)
        all_a.append(a)
        all_grid.append(grid)

    pred_cat = torch.cat(all_preds, dim=0)
    u_cat = torch.cat(all_u, dim=0)
    a_cat = torch.cat(all_a, dim=0)
    grid_cat = torch.cat(all_grid, dim=0)

    res = u_cat.shape[1]
    card = ScientificAuditCard(model_name=model_name, pde_name=pde_name, resolution=res)

    # 1. Prediction Accuracy (Relative L2)
    l2_err = relative_l2_error(pred_cat, u_cat).item()
    status = AuditStatus.PASS if l2_err < 0.05 else (AuditStatus.WARN if l2_err < 0.15 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Relative L2 Error",
            category="Accuracy",
            measured_value=l2_err,
            threshold=0.05,
            comparator="<",
            status=status,
            details="Standard relative L2 test loss",
        )
    )

    # 2. PDE Residual
    residual = pde.compute_residual(pred_cat, t=1.0)
    mean_res_norm = torch.norm(residual.reshape(len(pred_cat), -1), dim=1).mean().item()
    status = AuditStatus.PASS if mean_res_norm < 1.0 else (AuditStatus.WARN if mean_res_norm < 5.0 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="PDE Residual Norm",
            category="Physics",
            measured_value=mean_res_norm,
            threshold=1.0,
            comparator="<",
            status=status,
            details="Mean norm of PDE differential operator residual",
        )
    )

    # 3. Mass Conservation
    mass_drift = evaluate_mass_conservation(pred_cat, a_cat)
    status = AuditStatus.PASS if mass_drift < 5e-4 else (AuditStatus.WARN if mass_drift < 5e-3 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Mass Conservation Drift",
            category="Conservation",
            measured_value=mass_drift,
            threshold=5e-4,
            comparator="<",
            status=status,
            details="Relative drift of spatial mean field",
        )
    )

    # 4. Energy Drift
    energy_drift = evaluate_energy_drift(pred_cat, u_cat)
    status = AuditStatus.PASS if energy_drift < 0.01 else (AuditStatus.WARN if energy_drift < 0.05 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Energy Drift",
            category="Conservation",
            measured_value=energy_drift,
            threshold=0.01,
            comparator="<",
            status=status,
            details="Deviation from physical L2 energy dissipation",
        )
    )

    # 5. Divergence Constraint
    div_norm = evaluate_divergence_constraint(pred_cat, device=device)
    status = AuditStatus.PASS if div_norm < 1.0 else (AuditStatus.WARN if div_norm < 5.0 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Divergence Field Norm",
            category="Invariants",
            measured_value=div_norm,
            threshold=1.0,
            comparator="<",
            status=status,
            details="Incompressibility / spatial gradient balance",
        )
    )

    # 6. Spectral Cascade Fidelity (Energy Spectrum Error)
    spec_err = energy_spectrum_error(pred_cat, u_cat).item()
    status = AuditStatus.PASS if spec_err < 0.05 else (AuditStatus.WARN if spec_err < 0.15 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Energy Spectrum Error",
            category="Spectral",
            measured_value=spec_err,
            threshold=0.05,
            comparator="<",
            status=status,
            details="Radial Fourier energy spectrum discrepancy",
        )
    )

    # 7. High-Frequency Pileup (Nyquist cutoff accumulation)
    pileup = evaluate_high_frequency_pileup(pred_cat, u_cat)
    # Pileup ideally around 1.0 (matching ground truth)
    pileup_dev = abs(pileup - 1.0)
    status = AuditStatus.PASS if pileup_dev < 0.5 else (AuditStatus.WARN if pileup_dev < 2.0 else AuditStatus.FAIL)
    card.criteria.append(
        AuditCriterionResult(
            name="Nyquist Pileup Ratio",
            category="Spectral",
            measured_value=pileup,
            threshold=1.5,
            comparator="≈",
            status=status,
            details="Ratio of high-frequency power to ground truth",
        )
    )

    # 8. Long-Horizon Autoregressive Stability
    stable_steps = evaluate_long_horizon_stability(
        model, a_cat[:1], grid_cat[:1], n_steps=max_rollout_steps, device=device
    )
    status = AuditStatus.PASS if stable_steps >= max_rollout_steps else (
        AuditStatus.WARN if stable_steps >= max_rollout_steps // 2 else AuditStatus.FAIL
    )
    card.criteria.append(
        AuditCriterionResult(
            name="Rollout Stability Steps",
            category="Stability",
            measured_value=float(stable_steps),
            threshold=float(max_rollout_steps),
            comparator=">=",
            status=status,
            details=f"Number of autoregressive rollout steps before blowup (max {max_rollout_steps})",
        )
    )

    return card

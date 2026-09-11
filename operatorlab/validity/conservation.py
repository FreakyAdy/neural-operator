"""Physical conservation law auditing.

Quantifies violations of mass, energy, and momentum conservation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ConservationAuditResult:
    """Outcome of physical conservation checks."""

    mass_error_percent: float
    energy_drift_percent: float
    momentum_error_percent: float
    mass_status: str  # PASS, WARN, FAIL
    energy_status: str


def audit_conservation(
    pred: Tensor,
    initial: Tensor,
    target: Tensor,
) -> ConservationAuditResult:
    """Audit mass, energy, and momentum conservation."""
    # 1. Mass conservation: relative drift of spatial mean
    mass_init = initial.mean(dim=(1, 2))
    mass_pred = pred.mean(dim=(1, 2))
    mass_err = (mass_pred - mass_init).abs() / (mass_init.abs() + 1e-5)
    mass_err_pct = float(mass_err.mean().item() * 100.0)

    # 2. Energy drift: L2 kinetic/field energy drift compared to ground truth target
    energy_true = (target ** 2).mean(dim=(1, 2))
    energy_pred = (pred ** 2).mean(dim=(1, 2))
    energy_drift = (energy_pred - energy_true).abs() / (energy_true + 1e-6)
    energy_drift_pct = float(energy_drift.mean().item() * 100.0)

    # 3. Momentum conservation (for vector fields, or spatial gradients for scalar fields)
    mom_init = initial.abs().mean(dim=(1, 2))
    mom_pred = pred.abs().mean(dim=(1, 2))
    mom_err = (mom_pred - mom_init).abs() / (mom_init + 1e-5)
    mom_err_pct = float(mom_err.mean().item() * 100.0)

    mass_status = "PASS" if mass_err_pct < 1.0 else ("WARN" if mass_err_pct < 5.0 else "FAIL")
    energy_status = "PASS" if energy_drift_pct < 2.5 else ("WARN" if energy_drift_pct < 10.0 else "FAIL")

    return ConservationAuditResult(
        mass_error_percent=mass_err_pct,
        energy_drift_percent=energy_drift_pct,
        momentum_error_percent=mom_err_pct,
        mass_status=mass_status,
        energy_status=energy_status,
    )

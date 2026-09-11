"""Numerical stability and boundedness auditing.

Ensures neural operator predictions remain mathematically bounded, finite,
and free from finite-time blowup.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class StabilityAuditResult:
    """Outcome of numerical stability auditing."""

    is_finite: bool
    max_magnitude: float
    max_reference_magnitude: float
    norm_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def audit_stability(
    pred: Tensor,
    reference: Tensor,
    max_allowed_growth: float = 5.0,
) -> StabilityAuditResult:
    """Audit boundedness and numerical stability."""
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        return StabilityAuditResult(
            is_finite=False,
            max_magnitude=float("nan"),
            max_reference_magnitude=reference.abs().max().item(),
            norm_ratio=float("inf"),
            status="FAIL",
            message="Numerical explosion (NaN/Inf detected)",
        )

    pred_max = pred.abs().max().item()
    ref_max = max(reference.abs().max().item(), 1e-6)
    ratio = pred_max / ref_max

    if ratio <= max_allowed_growth:
        status = "PASS"
        msg = f"Bounded (max amplitude {pred_max:.3f} <= {max_allowed_growth * ref_max:.3f})"
    elif ratio <= max_allowed_growth * 2.0:
        status = "WARN"
        msg = f"Mild unbounded growth (ratio {ratio:.2f}x)"
    else:
        status = "FAIL"
        msg = f"Unphysical growth / divergence (ratio {ratio:.2f}x)"

    return StabilityAuditResult(
        is_finite=True,
        max_magnitude=pred_max,
        max_reference_magnitude=ref_max,
        norm_ratio=ratio,
        status=status,
        message=msg,
    )

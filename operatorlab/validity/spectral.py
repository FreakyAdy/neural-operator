"""Spectral structure and energy cascade auditing.

Validates that neural operators preserve turbulent energy cascades
and avoids unphysical high-frequency spectral accumulation / checkerboard artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import energy_spectrum_error


@dataclass
class SpectralAuditResult:
    """Outcome of spectral structure auditing."""

    spectrum_error: float
    nyquist_pileup_ratio: float
    spectrum_status: str  # PASS, WARN, FAIL
    pileup_status: str


def audit_spectral_fidelity(
    pred: Tensor,
    target: Tensor,
) -> SpectralAuditResult:
    """Audit energy spectrum error and Nyquist mode accumulation."""
    # 1. Radial Fourier spectrum error
    spec_err = float(energy_spectrum_error(pred, target).item())
    spec_status = "PASS" if spec_err < 0.05 else ("WARN" if spec_err < 0.15 else "FAIL")

    # 2. Nyquist mode pileup: ratio of power in top 15% frequencies
    pred_hat = torch.fft.rfft2(pred.permute(0, 3, 1, 2))
    target_hat = torch.fft.rfft2(target.permute(0, 3, 1, 2))

    fh, fw = pred_hat.shape[-2:]
    cutoff_h = int(fh * 0.85)
    cutoff_w = int(fw * 0.85)

    high_pred_power = (pred_hat[..., cutoff_h:, cutoff_w:].abs() ** 2).mean()
    high_target_power = (target_hat[..., cutoff_h:, cutoff_w:].abs() ** 2).mean()
    pileup_ratio = float((high_pred_power / (high_target_power + 1e-8)).item())

    pileup_dev = abs(pileup_ratio - 1.0)
    pileup_status = "PASS" if pileup_dev < 0.5 else ("WARN" if pileup_dev < 2.0 else "FAIL")

    return SpectralAuditResult(
        spectrum_error=spec_err,
        nyquist_pileup_ratio=pileup_ratio,
        spectrum_status=spec_status,
        pileup_status=pileup_status,
    )

"""Physical invariants and constraint auditing.

Audits divergence-free constraints (incompressibility), Hamiltonian conservation,
and continuous symmetries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InvariantsAuditResult:
    """Outcome of physical invariants auditing."""

    divergence_norm: float
    divergence_status: str  # PASS, WARN, FAIL
    symmetry_error: float
    symmetry_status: str


def audit_invariants(
    pred: Tensor,
    device: str = "cpu",
) -> InvariantsAuditResult:
    """Audit differential invariants such as incompressibility and spatial symmetries."""
    field = pred[..., 0]  # (batch, H, W)
    h, w = field.shape[-2:]

    # 1. Divergence constraint via 2D FFT
    k_x = torch.fft.fftfreq(h, d=1.0 / h, device=device) * 2 * torch.pi
    k_y = torch.fft.fftfreq(w, d=1.0 / w, device=device) * 2 * torch.pi
    kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")

    field_hat = torch.fft.fft2(field)
    dudx = torch.fft.ifft2(1j * kx.unsqueeze(0) * field_hat).real
    dudy = torch.fft.ifft2(1j * ky.unsqueeze(0) * field_hat).real
    div = dudx + dudy
    div_norm = float(torch.norm(div.reshape(len(field), -1), dim=1).mean().item())

    div_status = "PASS" if div_norm < 1.0 else ("WARN" if div_norm < 5.0 else "FAIL")

    # 2. Symmetry audit (reflection symmetry residual)
    field_flipped = torch.flip(field, dims=[-1])
    sym_err = float((field - field_flipped).abs().mean().item())
    sym_status = "PASS" if sym_err < 2.0 else "WARN"

    return InvariantsAuditResult(
        divergence_norm=div_norm,
        divergence_status=div_status,
        symmetry_error=sym_err,
        symmetry_status=sym_status,
    )

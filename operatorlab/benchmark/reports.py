"""Operator Generalization and Robustness Audit Reports.

Formats executive scientific audit cards matching the OperatorLab research engine standard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


import sys


def _safe_encode_text(text: str) -> str:
    """Ensure text can be safely rendered on terminal streams with limited codecs."""
    try:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return (
            text.replace("────────────────────────────────────────", "----------------------------------------")
            .replace("─", "-")
            .replace("→", "->")
            .replace("ν", "nu")
            .replace("α", "alpha")
            .replace("σ", "sigma")
            .replace("×", "x")
        )


@dataclass
class GeneralizationAuditReport:
    """Consolidated audit report across all shift dimensions and physical validity checks."""

    model_name: str
    pde_name: str
    base_res: int
    target_res: int
    resolution_status: str     # PASS, WARN, FAIL
    parameter_desc: str
    parameter_status: str
    boundary_desc: str
    boundary_status: str
    geometry_desc: str
    geometry_status: str
    noise_desc: str
    noise_status: str
    rollout_desc: str
    rollout_status: str
    mass_error_pct: float
    energy_drift_pct: float
    reliability_score: int     # 0 to 100

    def format_audit_card(self, force_ascii: bool = False) -> str:
        """Format the exact OPERATOR GENERALIZATION AUDIT layout."""
        lines = [
            "OPERATOR GENERALIZATION AUDIT",
            "────────────────────────────────────────",
            "",
            "Resolution OOD",
            f"{self.base_res} → {self.target_res:<28} {self.resolution_status}",
            "",
            "Parameter OOD",
            f"{self.parameter_desc:<32} {self.parameter_status}",
            "",
            "Boundary OOD",
            f"{self.boundary_desc:<32} {self.boundary_status}",
            "",
            "Geometry OOD",
            f"{self.geometry_desc:<32} {self.geometry_status}",
            "",
            "Input Noise",
            f"{self.noise_desc:<32} {self.noise_status}",
            "",
            "Long-Horizon Rollout",
            f"{self.rollout_desc:<32} {self.rollout_status}",
            "",
            "Physics Conservation",
            f"{'Mass error':<32} {self.mass_error_pct:.2f}%",
            f"{'Energy drift':<32} {self.energy_drift_pct:.2f}%",
            "",
            "Overall Scientific Reliability",
            f"{self.reliability_score:>38}/100",
            "────────────────────────────────────────",
        ]
        card = "\n".join(lines)
        if force_ascii:
            return (
                card.replace("────────────────────────────────────────", "----------------------------------------")
                .replace("─", "-")
                .replace("→", "->")
                .replace("ν", "nu")
                .replace("α", "alpha")
                .replace("σ", "sigma")
                .replace("×", "x")
            )
        return _safe_encode_text(card)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

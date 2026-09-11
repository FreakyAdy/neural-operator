"""OperatorArena Leaderboard data models and presentation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LeaderboardEntry:
    """Entry for a single model in OperatorArena."""

    rank: int
    model_name: str
    arena_score: float        # Composite 0-100 score
    relative_l2: float        # In-domain accuracy
    ood_score: float          # OOD generalization score (0-100)
    robustness_score: float   # Stress testing score (0-100)
    validity_verdict: str     # PASS, WARN, FAIL
    mass_conservation: float  # Conservation violation
    latency_ms: float         # Inference speed per sample
    parameters: int           # Parameter count


class OperatorArenaLeaderboard:
    """Manages and displays competitive standings in OperatorArena."""

    def __init__(self, pde_name: str = "NavierStokes2D", resolution: int = 64) -> None:
        self.pde_name = pde_name
        self.resolution = resolution
        self.entries: list[LeaderboardEntry] = []

    def add_entry(self, entry: LeaderboardEntry) -> None:
        self.entries.append(entry)
        self.sort()

    def sort(self) -> None:
        """Sort entries by arena_score descending and reassign ranks."""
        self.entries.sort(key=lambda e: e.arena_score, reverse=True)
        for i, entry in enumerate(self.entries, start=1):
            entry.rank = i

    def terminal_table(self) -> str:
        """Format an executive ASCII / Unicode table."""
        lines = [
            f"╔{'═' * 106}╗",
            f"║ OPERATOR ARENA LEADERBOARD — {self.pde_name} ({self.resolution}×{self.resolution})".ljust(107) + "║",
            f"╠{'═' * 106}╣",
            f"║ {'Rank':<4} | {'Model':<14} | {'Arena Score':<11} | {'L2 Error':<9} | {'OOD Score':<9} | {'Robustness':<10} | {'Scientific':<10} | {'Latency':<8} | {'Params':<9} ║",
            f"╟{'─' * 6}┼{'─' * 16}┼{'─' * 13}┼{'─' * 11}┼{'─' * 11}┼{'─' * 12}┼{'─' * 12}┼{'─' * 10}┼{'─' * 11}╢",
        ]

        for e in self.entries:
            params_str = f"{e.parameters / 1e3:.1f}K" if e.parameters < 1e6 else f"{e.parameters / 1e6:.2f}M"
            latency_str = f"{e.latency_ms:.2f}ms"
            lines.append(
                f"║ #{e.rank:<3} | {e.model_name:<14} | {e.arena_score:<11.1f} | {e.relative_l2:<9.4f} | {e.ood_score:<9.1f} | {e.robustness_score:<10.1f} | {e.validity_verdict:<10} | {latency_str:<8} | {params_str:<9} ║"
            )

        lines.append(f"╚{'═' * 106}╝")
        return "\n".join(lines)

    def markdown_table(self) -> str:
        """Format markdown table for documentation / GitHub export."""
        lines = [
            f"### OperatorArena Leaderboard: {self.pde_name} ({self.resolution}×{self.resolution})",
            "",
            "| Rank | Model | Arena Score | Relative L2 | OOD Score | Robustness | Scientific Validity | Latency | Parameters |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for e in self.entries:
            params_str = f"{e.parameters / 1e3:.1f}k" if e.parameters < 1e6 else f"{e.parameters / 1e6:.2f}M"
            lines.append(
                f"| **#{e.rank}** | **{e.model_name}** | **{e.arena_score:.1f}** | {e.relative_l2:.4f} | {e.ood_score:.1f} | {e.robustness_score:.1f} | `{e.validity_verdict}` | {e.latency_ms:.2f} ms | {params_str} |"
            )
        return "\n".join(lines)

    def save_json(self, path: Path | str) -> None:
        """Save leaderboard to JSON file."""
        data = {
            "pde_name": self.pde_name,
            "resolution": self.resolution,
            "entries": [asdict(e) for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

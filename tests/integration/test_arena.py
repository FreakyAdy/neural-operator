"""Integration tests for OperatorArena benchmark and leaderboard."""

from __future__ import annotations

import pytest
import torch

from operatorlab.arena.benchmark import ArenaBenchmark
from operatorlab.arena.leaderboard import LeaderboardEntry, OperatorArenaLeaderboard
from operatorlab.models.fno import FNO2d
from operatorlab.models.tfno import TFNO2d
from operatorlab.physics.heat import HeatEquation2D


class TestOperatorArena:
    def test_leaderboard_sorting_and_rendering(self) -> None:
        leaderboard = OperatorArenaLeaderboard(pde_name="Heat2D", resolution=32)
        leaderboard.add_entry(
            LeaderboardEntry(
                rank=0,
                model_name="ModelB",
                arena_score=75.5,
                relative_l2=0.03,
                ood_score=70.0,
                robustness_score=80.0,
                validity_verdict="WARN",
                mass_conservation=1e-4,
                latency_ms=1.2,
                parameters=50000,
            )
        )
        leaderboard.add_entry(
            LeaderboardEntry(
                rank=0,
                model_name="ModelA",
                arena_score=88.2,
                relative_l2=0.01,
                ood_score=85.0,
                robustness_score=90.0,
                validity_verdict="PASS",
                mass_conservation=1e-5,
                latency_ms=0.8,
                parameters=20000,
            )
        )

        assert leaderboard.entries[0].model_name == "ModelA"
        assert leaderboard.entries[0].rank == 1
        assert leaderboard.entries[1].model_name == "ModelB"
        assert leaderboard.entries[1].rank == 2

        t_table = leaderboard.terminal_table()
        assert "OPERATOR ARENA LEADERBOARD" in t_table
        assert "ModelA" in t_table

        m_table = leaderboard.markdown_table()
        assert "| **#1** | **ModelA** |" in m_table

    def test_arena_benchmark_smoke_run(self) -> None:
        pde = HeatEquation2D(alpha=0.01, T=0.1)
        arena = ArenaBenchmark(pde=pde, resolution=16, n_eval_samples=4, device="cpu")

        models = {
            "FNO": FNO2d(modes1=4, modes2=4, width=16, n_layers=2),
            "TFNO": TFNO2d(modes1=4, modes2=4, width=16, n_layers=2, rank=4),
        }

        leaderboard = arena.run(models)
        assert len(leaderboard.entries) == 2
        assert leaderboard.entries[0].rank == 1
        assert leaderboard.entries[1].rank == 2
        assert leaderboard.entries[0].arena_score > 0

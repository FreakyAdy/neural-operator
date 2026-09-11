"""OperatorArena — Standardized scientific benchmark and leaderboard for neural operators."""

from __future__ import annotations

from operatorlab.arena.benchmark import ArenaBenchmark, ArenaModelMetrics
from operatorlab.arena.leaderboard import LeaderboardEntry, OperatorArenaLeaderboard

__all__ = [
    "ArenaBenchmark",
    "ArenaModelMetrics",
    "LeaderboardEntry",
    "OperatorArenaLeaderboard",
]

"""Competitive ranking algorithms for OperatorArena."""

from __future__ import annotations

from typing import Dict, List

from operatorlab.benchmark.reports import GeneralizationAuditReport


def rank_models_by_reliability(
    reports: List[GeneralizationAuditReport],
) -> List[GeneralizationAuditReport]:
    """Sort models by overall scientific reliability score in descending order."""
    return sorted(reports, key=lambda r: r.reliability_score, reverse=True)

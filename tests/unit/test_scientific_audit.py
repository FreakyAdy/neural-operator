"""Unit tests for Scientific Validity and Physical Invariants Audit Layer."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from operatorlab.data.datasets import InMemoryDataset
from operatorlab.evaluation.scientific_audit import (
    AuditStatus,
    ScientificAuditCard,
    evaluate_divergence_constraint,
    evaluate_energy_drift,
    evaluate_high_frequency_pileup,
    evaluate_long_horizon_stability,
    evaluate_mass_conservation,
    run_scientific_audit,
)
from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D


class TestScientificMetrics:
    def test_evaluate_mass_conservation(self) -> None:
        init = torch.ones(2, 16, 16, 1) * 2.0
        # Exactly identical output -> drift 0
        drift_zero = evaluate_mass_conservation(init, init)
        assert abs(drift_zero) < 1e-5

        # Perturbed output -> drift > 0
        drift_perturbed = evaluate_mass_conservation(init * 1.1, init)
        assert 0.08 < drift_perturbed < 0.12

    def test_evaluate_energy_drift(self) -> None:
        target = torch.ones(2, 16, 16, 1)
        pred = target * 1.05
        drift = evaluate_energy_drift(pred, target)
        assert drift > 0.0

    def test_evaluate_divergence_constraint(self) -> None:
        field = torch.randn(2, 16, 16, 1)
        div = evaluate_divergence_constraint(field)
        assert div >= 0.0

    def test_evaluate_high_frequency_pileup(self) -> None:
        target = torch.randn(2, 16, 16, 1)
        ratio = evaluate_high_frequency_pileup(target, target)
        assert 0.95 < ratio < 1.05

    def test_evaluate_long_horizon_stability(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()
        a = torch.randn(1, 16, 16, 1)
        steps = evaluate_long_horizon_stability(model, a, None, n_steps=5)
        assert 0 <= steps <= 5


class TestScientificAuditCard:
    def test_audit_card_execution(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()
        pde = HeatEquation2D(alpha=0.01, T=0.1)

        # Generate small dataset
        data = pde.generate_dataset(n_samples=4, resolution=16, seed=42)
        ds = InMemoryDataset(data["a"], data["u"], normalize=False)
        loader = DataLoader(ds, batch_size=2)

        card = run_scientific_audit(model, pde, loader, device="cpu", max_rollout_steps=5)
        assert isinstance(card, ScientificAuditCard)
        assert len(card.criteria) >= 7
        assert card.overall_verdict in [
            "PHYSICALLY VALID (All Constraints Respected)",
            "CONDITIONALLY VALID (Warnings Present)",
            "PHYSICALLY INVALID (Critical Violations)",
        ]

        summary = card.summary_card()
        assert "Scientific Validity Audit" in summary
        assert "OVERALL VERDICT" in summary
        assert "Mass Conservation" in summary

        d = card.to_dict()
        assert "overall_verdict" in d
        assert len(d["criteria"]) == len(card.criteria)

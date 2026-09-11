"""Unit tests for the operatorlab.benchmark subsystem."""

from __future__ import annotations

import pytest
import torch

from operatorlab.benchmark.ood import run_generalization_audit
from operatorlab.benchmark.ranking import rank_models_by_reliability
from operatorlab.benchmark.reports import GeneralizationAuditReport
from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D


class TestBenchmarkSubsystem:
    def test_run_generalization_audit(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()
        pde = HeatEquation2D(alpha=0.01, T=0.1)

        report = run_generalization_audit(
            model=model,
            pde=pde,
            base_res=16,
            target_res=24,
            n_samples=4,
            device="cpu",
        )

        assert isinstance(report, GeneralizationAuditReport)
        assert report.resolution_status in ["PASS", "WARN", "FAIL"]
        assert report.parameter_status in ["PASS", "WARN", "FAIL"]
        assert report.boundary_status in ["PASS", "WARN", "FAIL"]
        assert report.geometry_status in ["PASS", "WARN", "FAIL"]
        assert report.noise_status in ["PASS", "WARN", "FAIL"]
        assert report.rollout_status in ["PASS", "WARN", "FAIL"]
        assert 0 <= report.reliability_score <= 100

        card = report.format_audit_card()
        assert "OPERATOR GENERALIZATION AUDIT" in card
        assert "Resolution OOD" in card
        assert "Parameter OOD" in card
        assert "Boundary OOD" in card
        assert "Geometry OOD" in card
        assert "Input Noise" in card
        assert "Long-Horizon Rollout" in card
        assert "Physics Conservation" in card
        assert "Overall Scientific Reliability" in card

    def test_ranking_by_reliability(self) -> None:
        r1 = GeneralizationAuditReport(
            model_name="ModelA", pde_name="Heat", base_res=16, target_res=32,
            resolution_status="PASS", parameter_desc="p", parameter_status="PASS",
            boundary_desc="b", boundary_status="PASS", geometry_desc="g", geometry_status="PASS",
            noise_desc="n", noise_status="PASS", rollout_desc="r", rollout_status="PASS",
            mass_error_pct=0.1, energy_drift_pct=0.5, reliability_score=95,
        )
        r2 = GeneralizationAuditReport(
            model_name="ModelB", pde_name="Heat", base_res=16, target_res=32,
            resolution_status="FAIL", parameter_desc="p", parameter_status="FAIL",
            boundary_desc="b", boundary_status="FAIL", geometry_desc="g", geometry_status="FAIL",
            noise_desc="n", noise_status="FAIL", rollout_desc="r", rollout_status="FAIL",
            mass_error_pct=5.0, energy_drift_pct=15.0, reliability_score=40,
        )

        ranked = rank_models_by_reliability([r2, r1])
        assert ranked[0].model_name == "ModelA"
        assert ranked[1].model_name == "ModelB"

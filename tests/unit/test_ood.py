"""Unit tests for OOD Operator Generalization Suite."""

from __future__ import annotations

import pytest
import torch

from operatorlab.evaluation.ood import (
    OODGeneralizationReport,
    OODShiftType,
    run_ood_generalization_benchmark,
    warp_grid_coordinates,
)
from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D


class TestOODGeneralization:
    def test_warp_grid_coordinates_shape(self) -> None:
        grid = warp_grid_coordinates(resolution=16, distortion_strength=0.05)
        assert grid.shape == (1, 16, 16, 2)
        # Check boundary bounds
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_ood_benchmark_smoke(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        pde = HeatEquation2D(alpha=0.01, T=0.1)

        report = run_ood_generalization_benchmark(
            model=model,
            pde=pde,
            base_resolution=16,
            n_samples=4,
            device="cpu",
            target_resolutions=[24],
            parameter_multipliers=[1.5],
            forcing_multipliers=[2.0],
        )

        assert isinstance(report, OODGeneralizationReport)
        assert len(report.cases) >= 5
        assert report.generalization_score > 0
        assert report.mean_degradation_ratio >= 0.0

        table = report.summary_table()
        assert "OOD OPERATOR GENERALIZATION REPORT" in table
        assert "Warped Coordinates" in table

        d = report.to_dict()
        assert "generalization_score" in d
        assert len(d["cases"]) == len(report.cases)

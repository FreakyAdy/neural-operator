"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from operatorlab.evaluation.metrics import (
    energy_spectrum_error,
    linf_error,
    relative_h1,
    relative_l2,
    speedup_vs_solver,
)


class TestMetrics:
    def test_relative_l2_exact(self) -> None:
        target = torch.randn(2, 16, 16, 1)
        pred = target.clone()
        err = relative_l2(pred, target)
        assert err.item() == pytest.approx(0.0, abs=1e-6)

    def test_relative_h1_exact(self) -> None:
        target = torch.randn(2, 16, 16, 1)
        pred = target.clone()
        err = relative_h1(pred, target)
        assert err.item() == pytest.approx(0.0, abs=1e-5)

    def test_linf_error(self) -> None:
        target = torch.zeros(2, 16, 16, 1)
        pred = torch.zeros(2, 16, 16, 1)
        pred[0, 5, 5, 0] = 2.5
        assert linf_error(pred, target).item() == pytest.approx(2.5, abs=1e-5)

    def test_energy_spectrum_error(self) -> None:
        target = torch.randn(2, 16, 16, 1)
        pred = target.clone()
        err = energy_spectrum_error(pred, target)
        assert err.item() == pytest.approx(0.0, abs=1e-5)

    def test_speedup_vs_solver(self) -> None:
        speedup = speedup_vs_solver(model_time_ms=10.0, solver_time_ms=500.0)
        assert speedup == pytest.approx(50.0)

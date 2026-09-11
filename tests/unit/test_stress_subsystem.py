"""Unit tests for the operatorlab.stress subsystem."""

from __future__ import annotations

import pytest
import torch

from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D
from operatorlab.stress.boundary_shift import evaluate_boundary_stress
from operatorlab.stress.geometry_shift import evaluate_geometry_stress
from operatorlab.stress.long_horizon import evaluate_long_horizon_stress
from operatorlab.stress.noise import evaluate_noise_stress
from operatorlab.stress.parameter_shift import evaluate_parameter_stress
from operatorlab.stress.resolution import evaluate_resolution_stress


class TestStressSubsystem:
    @pytest.fixture
    def setup_model_and_pde(self):
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()
        pde = HeatEquation2D(alpha=0.01, T=0.1)
        return model, pde

    def test_resolution_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_resolution_stress(model, pde, base_res=16, target_res=24, n_samples=4, device="cpu")
        assert res.base_res == 16
        assert res.target_res == 24
        assert res.status in ["PASS", "WARN", "FAIL"]
        assert res.degradation_ratio >= 0.0

    def test_parameter_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_parameter_stress(model, pde, resolution=16, shift_multiplier=2.0, n_samples=4, device="cpu")
        assert res.parameter_name == "α"
        assert res.status in ["PASS", "WARN", "FAIL"]

    def test_geometry_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_geometry_stress(model, pde, resolution=16, distortion_strength=0.05, n_samples=4, device="cpu")
        assert res.status in ["PASS", "WARN", "FAIL"]

    def test_boundary_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_boundary_stress(model, pde, resolution=16, boundary_width=2, n_samples=4, device="cpu")
        assert res.boundary_condition == "Periodic → Dirichlet"
        assert res.status in ["PASS", "WARN", "FAIL"]

    def test_noise_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_noise_stress(model, pde, resolution=16, sigma=0.05, n_samples=4, device="cpu")
        assert res.noise_type == "Input Noise"
        assert res.status in ["PASS", "WARN", "FAIL"]

    def test_long_horizon_stress(self, setup_model_and_pde) -> None:
        model, pde = setup_model_and_pde
        res = evaluate_long_horizon_stress(model, pde, resolution=16, target_steps=5, device="cpu")
        assert 0 <= res.stable_steps <= 5
        assert res.status in ["PASS", "WARN", "FAIL"]

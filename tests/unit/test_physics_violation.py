"""Unit tests for physics violation analysis."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from operatorlab.data.datasets import InMemoryDataset
from operatorlab.evaluation.physics_violation import analyze_physics_violations
from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D


class TestPhysicsViolation:
    def test_analyze_physics_violations(self) -> None:
        pde = HeatEquation2D(alpha=0.01, T=0.5)
        data = pde.generate_dataset(n_samples=4, resolution=16, seed=42)

        ds = InMemoryDataset(data["a"], data["u"], normalize=True)
        loader = DataLoader(ds, batch_size=2)

        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        report = analyze_physics_violations(model, pde, loader, device="cpu")

        assert report.mean_residual_norm >= 0.0
        assert report.max_residual_norm >= 0.0
        assert "mean_energy" in report.conservation_violations
        assert report.energy_spectrum_error >= 0.0

"""Unit tests for Operator Robustness and Stress Testing Suite."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from operatorlab.data.datasets import InMemoryDataset
from operatorlab.evaluation.stress import (
    OperatorRobustnessReport,
    perturb_boundary_noise,
    perturb_coordinate_jitter,
    perturb_gaussian_noise,
    perturb_sensor_sparsity,
    perturb_shot_noise,
    perturb_spectral_truncation,
    run_stress_test,
)
from operatorlab.models.fno import FNO2d


class TestStressTransforms:
    def test_gaussian_noise_preserves_shape(self) -> None:
        x = torch.randn(4, 16, 16, 1)
        noisy = perturb_gaussian_noise(x, sigma=0.05)
        assert noisy.shape == x.shape
        assert not torch.allclose(x, noisy)

    def test_sensor_sparsity(self) -> None:
        x = torch.ones(4, 16, 16, 1)
        sparse = perturb_sensor_sparsity(x, keep_ratio=0.5, mask_mode="zero")
        assert sparse.shape == x.shape
        zeros_ratio = (sparse == 0).float().mean().item()
        assert 0.3 < zeros_ratio < 0.7

    def test_spectral_truncation(self) -> None:
        x = torch.randn(4, 16, 16, 1)
        truncated = perturb_spectral_truncation(x, keep_ratio=0.5)
        assert truncated.shape == x.shape
        assert not torch.isnan(truncated).any()

    def test_coordinate_jitter(self) -> None:
        x = torch.linspace(0, 1, 16)
        gx, gy = torch.meshgrid(x, x, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(4, -1, -1, -1)
        jittered = perturb_coordinate_jitter(grid, jitter_std=0.02)
        assert jittered.shape == grid.shape
        assert jittered.min() >= 0.0
        assert jittered.max() <= 1.0

    def test_boundary_noise(self) -> None:
        x = torch.zeros(2, 16, 16, 1)
        b_noisy = perturb_boundary_noise(x, boundary_width=2, sigma=0.1)
        # Center should remain exactly 0
        center = b_noisy[:, 4:12, 4:12, :]
        assert torch.all(center == 0.0)

    def test_shot_noise(self) -> None:
        x = torch.zeros(2, 16, 16, 1)
        shot = perturb_shot_noise(x, p=0.1, scale=1.0)
        assert shot.shape == x.shape
        assert (shot != 0.0).any()


class TestStressPipeline:
    def test_run_stress_test(self) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()

        # Create synthetic dataset
        a = torch.randn(8, 16, 16, 1)
        u = torch.randn(8, 16, 16, 1)
        ds = InMemoryDataset(a, u, normalize=False)
        loader = DataLoader(ds, batch_size=4)

        report = run_stress_test(model, loader, device="cpu")
        assert isinstance(report, OperatorRobustnessReport)
        assert len(report.results) >= 8
        assert 0.0 <= report.robustness_score <= 100.0

        table = report.summary_table()
        assert "OPERATOR ROBUSTNESS REPORT" in table
        assert "Gaussian Noise" in table
        assert "Sensor Sparsity" in table

        d = report.to_dict()
        assert "robustness_score" in d
        assert len(d["results"]) == len(report.results)

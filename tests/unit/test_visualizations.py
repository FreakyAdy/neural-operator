"""Unit tests for visualization modules."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from operatorlab.models.fno import FNO2d
from operatorlab.physics.heat import HeatEquation2D
from operatorlab.visualization.fields import plot_field_comparison
from operatorlab.visualization.spectra import plot_error_spectrum
from operatorlab.visualization.trajectories import animate_rollout


class TestVisualizations:
    def test_plot_field_comparison(self, tmp_path: Path) -> None:
        a = torch.randn(2, 16, 16, 1)
        pred = torch.randn(2, 16, 16, 1)
        target = torch.randn(2, 16, 16, 1)

        save_file = tmp_path / "comparison.png"
        fig = plot_field_comparison(a, pred, target, save_path=save_file)
        assert fig is not None
        assert save_file.exists()

    def test_plot_error_spectrum(self, tmp_path: Path) -> None:
        pred = torch.randn(1, 16, 16, 1)
        target = torch.randn(1, 16, 16, 1)

        save_file = tmp_path / "spectrum.png"
        fig = plot_error_spectrum(pred, target, save_path=save_file)
        assert fig is not None
        assert save_file.exists()

    def test_animate_rollout(self, tmp_path: Path) -> None:
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        pde = HeatEquation2D(alpha=0.01, T=0.2)
        init = torch.randn(1, 16, 16, 1)

        save_file = tmp_path / "rollout.gif"
        animate_rollout(model, pde, init, T=0.2, n_steps=2, save_path=save_file)
        assert save_file.exists()

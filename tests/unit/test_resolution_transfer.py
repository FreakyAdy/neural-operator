"""Unit tests for resolution transfer across all architectures."""

from __future__ import annotations

import pytest
import torch


class TestResolutionTransfer:
    """Verify all models handle zero-shot resolution transfer."""

    def test_fno_resolution_transfer(self) -> None:
        """FNO handles 16→32→64 zero-shot."""
        from operatorlab.models.fno import FNO2d

        model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
        model.eval()

        for res in [16, 32, 64]:
            a = torch.randn(2, res, res, 1)
            with torch.no_grad():
                out = model(a)
            assert out.shape == (2, res, res, 1), f"Failed at resolution {res}"

    def test_tfno_resolution_transfer(self) -> None:
        """TFNO handles resolution transfer."""
        from operatorlab.models.tfno import TFNO2d

        model = TFNO2d(modes1=4, modes2=4, width=8, n_layers=2, rank=4)
        model.eval()

        for res in [16, 32, 64]:
            a = torch.randn(2, res, res, 1)
            with torch.no_grad():
                out = model(a)
            assert out.shape == (2, res, res, 1)

    def test_deeponet_resolution_transfer(self) -> None:
        """DeepONet handles resolution transfer via interpolated branch input."""
        from operatorlab.models.deeponet import DeepONet

        model = DeepONet(
            branch_input_dim=16 * 16,
            hidden_dim=16,
            n_basis=8,
            branch_depth=2,
            trunk_depth=2,
        )
        model.eval()

        for res in [16, 32]:
            a = torch.randn(2, res, res, 1)
            with torch.no_grad():
                out = model(a)
            assert out.shape == (2, res, res, 1), f"Failed at resolution {res}"

    def test_hybrid_resolution_transfer(self) -> None:
        """Hybrid operator handles resolution transfer."""
        from operatorlab.models.hybrid import HybridOperator

        model = HybridOperator(
            modes1=4, modes2=4, width=8,
            n_fno_layers=2, n_attention_layers=1,
            attention_heads=2, patch_size=4,
        )
        model.eval()

        for res in [16, 32]:
            a = torch.randn(2, res, res, 1)
            with torch.no_grad():
                out = model(a)
            assert out.shape == (2, res, res, 1)

    def test_all_report_resolution_support(self) -> None:
        """All models report supports_resolution_transfer()."""
        from operatorlab.models.fno import FNO2d
        from operatorlab.models.tfno import TFNO2d
        from operatorlab.models.deeponet import DeepONet
        from operatorlab.models.hybrid import HybridOperator
        from operatorlab.models.gno import GNO

        models = [
            FNO2d(modes1=4, modes2=4, width=8, n_layers=2),
            TFNO2d(modes1=4, modes2=4, width=8, n_layers=2, rank=4),
            DeepONet(branch_input_dim=64, hidden_dim=8, n_basis=4,
                     branch_depth=2, trunk_depth=2),
            HybridOperator(modes1=4, modes2=4, width=8,
                          n_fno_layers=2, n_attention_layers=1,
                          attention_heads=2, patch_size=4),
            GNO(input_dim=3, hidden_dim=8, n_layers=2, n_neighbors=4),
        ]

        for model in models:
            assert model.supports_resolution_transfer() is True, \
                f"{model.__class__.__name__} should support resolution transfer"

    def test_resolution_sweep_function(self) -> None:
        """Test the resolution_sweep evaluation pipeline and summary table."""
        from operatorlab.evaluation.resolution_sweep import resolution_sweep
        from operatorlab.models.fno import FNO2d
        from operatorlab.physics.heat import HeatEquation2D

        model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
        pde = HeatEquation2D(alpha=0.01, T=0.5)

        sweep = resolution_sweep(
            model=model,
            pde=pde,
            base_resolution=16,
            target_resolutions=[16, 32],
            n_test_samples=5,
            device="cpu",
        )

        assert len(sweep.results) == 2
        assert sweep.results[0].resolution == 16
        assert sweep.results[1].resolution == 32
        table_str = sweep.summary_table()
        assert "16×16" in table_str
        assert "32×32" in table_str

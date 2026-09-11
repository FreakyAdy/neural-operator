"""Unit tests for the Fourier Neural Operator (FNO)."""

from __future__ import annotations

import torch
import pytest

from operatorlab.models.fno import FNO2d, SpectralConv2d


class TestSpectralConv2d:
    """Tests for the spectral convolution layer."""

    def test_output_shape(self) -> None:
        """SpectralConv2d preserves spatial dims and maps channels correctly."""
        layer = SpectralConv2d(in_channels=16, out_channels=32, modes1=4, modes2=4)
        x = torch.randn(2, 16, 32, 32)  # (batch, in_ch, H, W)
        out = layer(x)
        assert out.shape == (2, 32, 32, 32)

    def test_output_is_real(self) -> None:
        """SpectralConv2d output must be real-valued (no imaginary part)."""
        layer = SpectralConv2d(in_channels=8, out_channels=8, modes1=4, modes2=4)
        x = torch.randn(2, 8, 16, 16)
        out = layer(x)
        assert not out.is_complex(), "SpectralConv2d output should be real-valued"

    def test_different_resolutions(self) -> None:
        """SpectralConv2d works at different spatial resolutions."""
        layer = SpectralConv2d(in_channels=8, out_channels=8, modes1=4, modes2=4)
        for res in [16, 32, 64]:
            x = torch.randn(2, 8, res, res)
            out = layer(x)
            assert out.shape == (2, 8, res, res)


class TestFNO2d:
    """Tests for the full FNO2d model."""

    def test_output_shape(self) -> None:
        """FNO2d produces correct output shape."""
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        a = torch.randn(4, 64, 64, 1)  # (batch, H, W, in_ch)
        out = model(a)
        assert out.shape == (4, 64, 64, 1)

    def test_output_shape_multichannel(self) -> None:
        """FNO2d works with multi-channel input/output."""
        model = FNO2d(
            modes1=4, modes2=4, width=16, n_layers=2,
            input_dim=3, output_dim=2,
        )
        a = torch.randn(4, 32, 32, 3)
        out = model(a)
        assert out.shape == (4, 32, 32, 2)

    def test_resolution_transfer(self) -> None:
        """FNO2d supports zero-shot resolution transfer."""
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        model.eval()

        # Train resolution
        a_32 = torch.randn(2, 32, 32, 1)
        # Higher resolution (zero-shot)
        a_64 = torch.randn(2, 64, 64, 1)

        with torch.no_grad():
            out_32 = model(a_32)
            out_64 = model(a_64)

        assert out_32.shape == (2, 32, 32, 1)
        assert out_64.shape == (2, 64, 64, 1)

    def test_supports_resolution_transfer_flag(self) -> None:
        """FNO2d reports resolution transfer support."""
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        assert model.supports_resolution_transfer() is True

    def test_count_parameters(self) -> None:
        """count_parameters returns a positive integer."""
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        n = model.count_parameters()
        assert isinstance(n, int)
        assert n > 0

    def test_custom_grid(self) -> None:
        """FNO2d accepts externally provided grid coordinates."""
        model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
        a = torch.randn(2, 16, 16, 1)
        grid = torch.rand(2, 16, 16, 2)  # custom grid
        out = model(a, grid=grid)
        assert out.shape == (2, 16, 16, 1)

    def test_backward_pass(self) -> None:
        """FNO2d supports backpropagation."""
        model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
        a = torch.randn(2, 16, 16, 1)
        out = model(a)
        loss = out.sum()
        loss.backward()

        # Check at least some gradients are non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients computed in backward pass"

    def test_get_config(self) -> None:
        """get_config returns expected keys."""
        model = FNO2d(modes1=8, modes2=8, width=32, n_layers=4)
        config = model.get_config()
        assert config["class"] == "FNO2d"
        assert config["modes1"] == 8
        assert config["width"] == 32
        assert config["resolution_transfer"] is True

"""Unit tests for Tensorized Fourier Neural Operator (TFNO)."""

from __future__ import annotations

import pytest
import torch

from operatorlab.models.tfno import TFNO2d, TuckerSpectralConv2d


class TestTuckerSpectralConv2d:
    def test_output_shape(self) -> None:
        conv = TuckerSpectralConv2d(in_channels=8, out_channels=8, modes1=4, modes2=4, rank=4)
        x = torch.randn(2, 8, 16, 16)
        out = conv(x)
        assert out.shape == (2, 8, 16, 16)

    def test_backward(self) -> None:
        conv = TuckerSpectralConv2d(in_channels=4, out_channels=4, modes1=4, modes2=4, rank=2)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestTFNO2d:
    def test_output_shape(self) -> None:
        model = TFNO2d(modes1=4, modes2=4, width=16, n_layers=2, rank=4)
        a = torch.randn(2, 16, 16, 1)
        out = model(a)
        assert out.shape == (2, 16, 16, 1)

    def test_parameter_count(self) -> None:
        model = TFNO2d(modes1=4, modes2=4, width=16, n_layers=2, rank=4)
        n_params = model.count_parameters()
        assert n_params > 0

    def test_resolution_transfer(self) -> None:
        model = TFNO2d(modes1=4, modes2=4, width=16, n_layers=2, rank=4)
        model.eval()
        a_16 = torch.randn(1, 16, 16, 1)
        a_32 = torch.randn(1, 32, 32, 1)
        with torch.no_grad():
            out_16 = model(a_16)
            out_32 = model(a_32)
        assert out_16.shape == (1, 16, 16, 1)
        assert out_32.shape == (1, 32, 32, 1)

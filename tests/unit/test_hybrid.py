"""Unit tests for Fourier-Attention Hybrid Operator."""

from __future__ import annotations

import pytest
import torch

from operatorlab.models.hybrid import HybridOperator, LocalAttention2d


class TestLocalAttention2d:
    def test_attention_shape(self) -> None:
        attn = LocalAttention2d(dim=16, n_heads=4, patch_size=4)
        x = torch.randn(2, 16, 16, 16)
        out = attn(x)
        assert out.shape == (2, 16, 16, 16)


class TestHybridOperator:
    def test_output_shape(self) -> None:
        model = HybridOperator(
            modes1=4,
            modes2=4,
            width=16,
            n_fno_layers=2,
            n_attention_layers=1,
            attention_heads=2,
            patch_size=4,
        )
        a = torch.randn(2, 16, 16, 1)
        out = model(a)
        assert out.shape == (2, 16, 16, 1)

    def test_count_parameters(self) -> None:
        model = HybridOperator(
            modes1=4,
            modes2=4,
            width=16,
            n_fno_layers=2,
            n_attention_layers=1,
        )
        assert model.count_parameters() > 0

    def test_resolution_transfer(self) -> None:
        model = HybridOperator(
            modes1=4,
            modes2=4,
            width=16,
            n_fno_layers=2,
            n_attention_layers=1,
            patch_size=4,
        )
        model.eval()
        a_16 = torch.randn(1, 16, 16, 1)
        a_32 = torch.randn(1, 32, 32, 1)
        with torch.no_grad():
            out_16 = model(a_16)
            out_32 = model(a_32)
        assert out_16.shape == (1, 16, 16, 1)
        assert out_32.shape == (1, 32, 32, 1)

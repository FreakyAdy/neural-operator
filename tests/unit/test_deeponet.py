"""Unit tests for DeepONet."""

from __future__ import annotations

import torch
import pytest

from operatorlab.models.deeponet import DeepONet


class TestDeepONet:
    """Tests for the Deep Operator Network."""

    def test_output_shape(self) -> None:
        model = DeepONet(
            branch_input_dim=32 * 32,
            hidden_dim=16,
            n_basis=8,
            branch_depth=2,
            trunk_depth=2,
        )
        a = torch.randn(4, 32, 32, 1)
        out = model(a)
        assert out.shape == (4, 32, 32, 1)

    def test_custom_grid(self) -> None:
        model = DeepONet(
            branch_input_dim=16 * 16,
            hidden_dim=16,
            n_basis=8,
            branch_depth=2,
            trunk_depth=2,
        )
        a = torch.randn(2, 16, 16, 1)
        grid = torch.rand(2, 16, 16, 2)
        out = model(a, grid=grid)
        assert out.shape == (2, 16, 16, 1)

    def test_backward(self) -> None:
        model = DeepONet(
            branch_input_dim=16 * 16,
            hidden_dim=8,
            n_basis=4,
            branch_depth=2,
            trunk_depth=2,
        )
        a = torch.randn(2, 16, 16, 1)
        out = model(a)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_count_parameters(self) -> None:
        model = DeepONet(
            branch_input_dim=64,
            hidden_dim=16,
            n_basis=8,
            branch_depth=2,
            trunk_depth=2,
        )
        assert model.count_parameters() > 0

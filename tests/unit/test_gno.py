"""Unit tests for Graph Neural Operator (GNO)."""

from __future__ import annotations

import pytest
import torch

from operatorlab.models.gno import GNO, GNOLayer


class TestGNOLayer:
    def test_layer_forward(self) -> None:
        layer = GNOLayer(hidden_dim=16, edge_dim=3)
        n_nodes = 25
        x = torch.randn(n_nodes, 16)
        # Create a simple ring graph
        src = torch.arange(n_nodes)
        dst = (torch.arange(n_nodes) + 1) % n_nodes
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.randn(n_nodes, 3)

        out = layer(x, edge_index, edge_attr)
        assert out.shape == (n_nodes, 16)


class TestGNO:
    def test_gno_forward_and_shape(self) -> None:
        model = GNO(input_dim=3, hidden_dim=16, output_dim=1, n_layers=2, n_neighbors=4)
        # 2 samples of 8x8 grid
        a = torch.randn(2, 8, 8, 1)
        out = model(a)
        assert out.shape == (2, 8, 8, 1)

    def test_gno_count_parameters(self) -> None:
        model = GNO(input_dim=3, hidden_dim=16, output_dim=1, n_layers=2, n_neighbors=4)
        assert model.count_parameters() > 0

    def test_gno_resolution_transfer(self) -> None:
        model = GNO(input_dim=3, hidden_dim=16, output_dim=1, n_layers=2, n_neighbors=4)
        model.eval()
        a_8 = torch.randn(1, 8, 8, 1)
        a_12 = torch.randn(1, 12, 12, 1)
        with torch.no_grad():
            out_8 = model(a_8)
            out_12 = model(a_12)
        assert out_8.shape == (1, 8, 8, 1)
        assert out_12.shape == (1, 12, 12, 1)

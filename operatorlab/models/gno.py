"""Graph Neural Operator (GNO).

Message-passing neural network over spatial point graphs for irregular meshes.
Each message-passing layer approximates one kernel integral operator layer.

Requires torch_geometric for production use. Provides a fallback implementation
using manual sparse message passing when torch_geometric is not available.

Reference: Li et al., "Neural Operator: Graph Kernel Network for Partial
Differential Equations" (2020).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from operatorlab.models.base import NeuralOperator

logger = logging.getLogger(__name__)

# Check for torch_geometric availability
_HAS_PYG = False
try:
    import torch_geometric  # type: ignore # noqa: F401
    from torch_geometric.nn import MessagePassing, knn_graph, radius_graph  # type: ignore # noqa: F401
    _HAS_PYG = True
except ImportError:
    pass


class GNOLayer(nn.Module):
    """Single GNO message-passing layer (fallback implementation).

    Approximates one kernel integral operator using neighbor aggregation.
    Edge features encode relative positions.

    Args:
        hidden_dim: Feature dimension.
        edge_dim: Edge feature dimension (relative position + distance).
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 3) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Message passing step.

        Args:
            x: Node features, shape (n_nodes, hidden_dim).
            edge_index: Edge indices, shape (2, n_edges).
            edge_attr: Edge features, shape (n_edges, edge_dim).

        Returns:
            Updated node features, shape (n_nodes, hidden_dim).
        """
        src, dst = edge_index  # (n_edges,)

        # Compute edge messages
        edge_features = self.edge_mlp(edge_attr)  # (n_edges, hidden_dim)
        messages = x[src] * edge_features           # (n_edges, hidden_dim)

        # Aggregate messages per destination node (mean)
        n_nodes = x.shape[0]
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        count = torch.zeros(n_nodes, 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1), dtype=torch.float))
        count = count.clamp(min=1)
        agg = agg / count

        # Update node features with residual
        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + out)


class GNO(NeuralOperator):
    """Graph Neural Operator for irregular meshes.

    Builds a kNN graph from spatial coordinates and applies
    message-passing layers to approximate kernel integration.

    Args:
        input_dim: Input feature dimension (field channels + coord channels).
        hidden_dim: Hidden feature dimension.
        output_dim: Output channels.
        n_layers: Number of message-passing layers.
        n_neighbors: Number of neighbors for kNN graph.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 4,
        n_neighbors: int = 16,
    ) -> None:
        if not _HAS_PYG:
            logger.warning(
                "torch_geometric not found. GNO will use fallback implementation. "
                "For best performance, install: pip install torch-geometric"
            )

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors

        # Node lifting
        self.lifting = nn.Linear(input_dim, hidden_dim)

        # Message-passing layers
        self.layers = nn.ModuleList([
            GNOLayer(hidden_dim, edge_dim=3)
            for _ in range(n_layers)
        ])

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_graph(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Build kNN graph from positions.

        Args:
            pos: Point positions, shape (n_points, 2).

        Returns:
            edge_index: (2, n_edges)
            edge_attr: (n_edges, 3) — relative position + distance
        """
        n = pos.shape[0]
        k = min(self.n_neighbors, n - 1)

        # Compute pairwise distances
        dist = torch.cdist(pos, pos)  # (n, n)
        _, idx = dist.topk(k + 1, dim=1, largest=False)  # (n, k+1) including self
        idx = idx[:, 1:]  # remove self-loops, shape (n, k)

        # Build edge index
        src = idx.reshape(-1)  # (n*k,)
        dst = torch.arange(n, device=pos.device).unsqueeze(1).expand(-1, k).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)  # (2, n*k)

        # Edge features: relative position + distance
        rel_pos = pos[src] - pos[dst]  # (n*k, 2)
        distance = rel_pos.norm(dim=-1, keepdim=True)  # (n*k, 1)
        edge_attr = torch.cat([rel_pos, distance], dim=-1)  # (n*k, 3)

        return edge_index, edge_attr

    def forward(
        self,
        a: Tensor,
        grid: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            a: Input, shape (batch, H, W, in_channels).
            grid: Coordinates, shape (batch, H, W, 2).

        Returns:
            Output, shape (batch, H, W, output_dim).
        """
        batch_size, h, w, c = a.shape

        if grid is None:
            grid = self.make_grid((h, w), a.device)
            grid = grid.expand(batch_size, -1, -1, -1)

        # Concatenate field values and coordinates
        x = torch.cat([a, grid], dim=-1)  # (batch, H, W, in_dim)

        outputs = []
        for b in range(batch_size):
            # Flatten to node list
            x_flat = x[b].reshape(-1, x.shape[-1])  # (n_nodes, in_dim)
            pos = grid[b].reshape(-1, 2)              # (n_nodes, 2)

            # Build graph (precomputed per-sample)
            edge_index, edge_attr = self._build_graph(pos)

            # Lift
            h_flat = self.lifting(x_flat)  # (n_nodes, hidden_dim)

            # Message passing
            for layer in self.layers:
                h_flat = layer(h_flat, edge_index, edge_attr)

            # Project
            out_flat = self.projection(h_flat)  # (n_nodes, output_dim)
            outputs.append(out_flat.reshape(h, w, self.output_dim))

        return torch.stack(outputs, dim=0)  # (batch, H, W, output_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def supports_resolution_transfer(self) -> bool:
        """GNO supports resolution transfer — graph is rebuilt at new resolution."""
        return True

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_neighbors": self.n_neighbors,
        })
        return config

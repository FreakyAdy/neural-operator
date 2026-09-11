"""Deep Operator Network (DeepONet).

Two-branch architecture:
    G(a)(x) = Σ_k branch_k(a) · trunk_k(x) + b

Branch net encodes the input function at fixed sensor points.
Trunk net encodes query coordinates where we evaluate the output.
Naturally supports irregular meshes (trunk net is pointwise).

Reference: Lu et al., "Learning nonlinear operators via DeepONet" (2021).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from operatorlab.models.base import NeuralOperator


class DeepONet(NeuralOperator):
    """Deep Operator Network.

    Args:
        branch_input_dim: Number of sensor points for the branch net.
        trunk_input_dim: Spatial dimension of query points (usually 2 for 2D).
        hidden_dim: Hidden layer width for both branch and trunk nets.
        output_dim: Number of output channels.
        branch_depth: Number of hidden layers in the branch net.
        trunk_depth: Number of hidden layers in the trunk net.
        n_basis: Number of basis functions (dot product dimension).
    """

    def __init__(
        self,
        branch_input_dim: int = 64 * 64,
        trunk_input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 1,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        n_basis: int = 128,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_basis = n_basis

        # Branch net: encodes the input function a(sensor_points) → basis coefficients
        branch_layers: list[nn.Module] = [nn.Linear(branch_input_dim, hidden_dim), nn.GELU()]
        for _ in range(branch_depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        branch_layers.append(nn.Linear(hidden_dim, n_basis * output_dim))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk net: encodes query coordinates x → basis values
        trunk_layers: list[nn.Module] = [nn.Linear(trunk_input_dim, hidden_dim), nn.GELU()]
        for _ in range(trunk_depth - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        trunk_layers.append(nn.Linear(hidden_dim, n_basis * output_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)

        # Bias
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        a: Tensor,
        grid: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            a: Input function values, shape (batch, H, W, in_channels).
            grid: Query coordinates, shape (batch, H, W, 2).
                  If None, constructs a uniform grid.

        Returns:
            Output field, shape (batch, H, W, output_dim).
        """
        batch_size, h, w, c = a.shape

        if grid is None:
            grid = self.make_grid((h, w), a.device)  # (1, H, W, 2)
            grid = grid.expand(batch_size, -1, -1, -1)

        # Branch: flatten input function to sensor values
        # (batch, H, W, C) → (batch, H*W*C) → (batch, n_basis * output_dim)
        a_flat = a.reshape(batch_size, -1)

        # If the sensor count doesn't match, interpolate
        if a_flat.shape[1] != self.branch_input_dim:
            a_flat = torch.nn.functional.interpolate(
                a_flat.unsqueeze(1),  # (batch, 1, sensors)
                size=self.branch_input_dim,
                mode="linear",
                align_corners=True,
            ).squeeze(1)  # (batch, branch_input_dim)

        branch_out = self.branch_net(a_flat)  # (batch, n_basis * output_dim)
        branch_out = branch_out.reshape(batch_size, self.n_basis, self.output_dim)
        # (batch, n_basis, output_dim)

        # Trunk: evaluate at query coordinates
        # (batch, H, W, 2) → (batch, H*W, 2)
        n_queries = h * w
        grid_flat = grid.reshape(batch_size, n_queries, self.trunk_input_dim)
        trunk_out = self.trunk_net(grid_flat)  # (batch, n_queries, n_basis * output_dim)
        trunk_out = trunk_out.reshape(batch_size, n_queries, self.n_basis, self.output_dim)
        # (batch, n_queries, n_basis, output_dim)

        # Dot product: sum over basis functions
        # branch: (batch, 1, n_basis, output_dim), trunk: (batch, n_queries, n_basis, output_dim)
        output = torch.einsum(
            "bko,bqko->bqo",
            branch_out,
            trunk_out,
        )  # (batch, n_queries, output_dim)

        # Add bias and reshape
        output = output + self.bias  # (batch, n_queries, output_dim)
        output = output.reshape(batch_size, h, w, self.output_dim)

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def supports_resolution_transfer(self) -> bool:
        """DeepONet supports resolution transfer — trunk queries any point."""
        return True

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "branch_input_dim": self.branch_input_dim,
            "trunk_input_dim": self.trunk_input_dim,
            "hidden_dim": self.hidden_dim,
            "n_basis": self.n_basis,
            "output_dim": self.output_dim,
        })
        return config

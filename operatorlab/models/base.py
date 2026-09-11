"""Abstract base class for all neural operator architectures.

Every model in OperatorLab must subclass NeuralOperator and implement
the forward() and count_parameters() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class NeuralOperator(nn.Module, ABC):
    """Base class for neural operators.

    A neural operator maps between function spaces:
        G : A → U
    where A and U are infinite-dimensional function spaces.

    All subclasses must implement:
        - forward(a, grid) → u
        - count_parameters() → int
    """

    @abstractmethod
    def forward(
        self,
        a: Tensor,
        grid: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the neural operator.

        Args:
            a: Input function values, shape (batch, *spatial_dims, in_channels).
            grid: Spatial coordinates, shape (batch, *spatial_dims, space_dim).
                  If None, the model constructs a uniform grid internally.

        Returns:
            Output function values, shape (batch, *spatial_dims, out_channels).
        """

    @abstractmethod
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""

    def supports_resolution_transfer(self) -> bool:
        """Return True if the model is resolution-independent by design.

        Models that are resolution-independent can be trained at one spatial
        resolution and evaluated at a different resolution without retraining.
        """
        return False

    def get_config(self) -> dict:
        """Return hyperparameter dict for experiment registry."""
        return {
            "class": self.__class__.__name__,
            "n_parameters": self.count_parameters(),
            "resolution_transfer": self.supports_resolution_transfer(),
        }

    @staticmethod
    def make_grid(spatial_dims: tuple[int, ...], device: torch.device) -> Tensor:
        """Construct a uniform coordinate grid on [0, 1]^d.

        Args:
            spatial_dims: Tuple of spatial dimension sizes, e.g. (64, 64).
            device: Torch device for the output tensor.

        Returns:
            Grid tensor of shape (1, *spatial_dims, d) where d = len(spatial_dims).
        """
        linspaces = [
            torch.linspace(0, 1, s, device=device) for s in spatial_dims
        ]
        meshes = torch.meshgrid(*linspaces, indexing="ij")
        # Stack and add batch dim: (1, *spatial_dims, d)
        grid = torch.stack(meshes, dim=-1).unsqueeze(0)
        return grid

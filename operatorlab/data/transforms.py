"""Data normalization transforms for neural operator training.

Provides:
    - UnitGaussianNormalizer: zero-mean, unit-variance normalization
"""

from __future__ import annotations

import torch
from torch import Tensor


class UnitGaussianNormalizer:
    """Zero-mean, unit-variance normalizer for field data.

    Computes normalization statistics from training data and applies/removes
    the transform. Stats are stored so they can be serialized with datasets.

    Args:
        eps: Small constant to avoid division by zero in variance.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
        self.mean: Tensor | None = None
        self.std: Tensor | None = None
        self._fitted = False

    def fit(self, x: Tensor) -> UnitGaussianNormalizer:
        """Compute mean and std from training data.

        Args:
            x: Training data, shape (n_samples, *spatial, channels).

        Returns:
            Self, for chaining.
        """
        # Average over all dims except the last (channel dim)
        reduce_dims = tuple(range(x.ndim - 1))
        self.mean = x.mean(dim=reduce_dims)    # (channels,)
        self.std = x.std(dim=reduce_dims)      # (channels,)
        self._fitted = True
        return self

    def encode(self, x: Tensor) -> Tensor:
        """Normalize x to zero mean, unit variance.

        Args:
            x: Input tensor, shape (..., channels).

        Returns:
            Normalized tensor.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted. Call .fit() first.")
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x: Tensor) -> Tensor:
        """Reverse normalization to recover physical-space values.

        Args:
            x: Normalized tensor, shape (..., channels).

        Returns:
            Denormalized tensor.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted. Call .fit() first.")
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)

    def state_dict(self) -> dict[str, Tensor | float]:
        """Serialize normalizer state for saving with datasets."""
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, Tensor | float]) -> None:
        """Restore normalizer state from a saved dict."""
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"]
        self._fitted = True

"""PyTorch Dataset wrappers for PDE data.

Supports:
    - Loading pre-generated HDF5 data
    - In-memory tensor datasets
    - On-the-fly resolution resampling
    - Automatic grid coordinate generation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from operatorlab.data.transforms import UnitGaussianNormalizer

logger = logging.getLogger(__name__)


class PDEDataset(Dataset):
    """Dataset wrapping pre-generated PDE data.

    Returns (a, u, grid) tuples where:
        a: initial/boundary condition, shape (*spatial, in_channels)
        u: target field, shape (*spatial, out_channels)
        grid: coordinate grid, shape (*spatial, spatial_dim)

    Args:
        path: Path to HDF5 file containing 'a' and 'u' arrays.
        split: Which split to load ('train', 'val', 'test').
        resolution: If set, resample fields to this resolution.
        normalize: Whether to apply zero-mean/unit-variance normalization.
        device: Device to load tensors to.
    """

    def __init__(
        self,
        path: Path,
        split: str = "train",
        resolution: Optional[int] = None,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        self.path = Path(path)
        self.split = split
        self.resolution = resolution
        self.normalize = normalize
        self.device = device

        self._load_data()

        if normalize:
            self._fit_normalizers()

    def _load_data(self) -> None:
        """Load data from HDF5 file."""
        import h5py

        with h5py.File(self.path, "r") as f:
            if self.split in f:
                group = f[self.split]
            else:
                group = f

            self.a = torch.tensor(
                np.array(group["a"]), dtype=torch.float32, device=self.device,
            )
            self.u = torch.tensor(
                np.array(group["u"]), dtype=torch.float32, device=self.device,
            )

        # Ensure channel dimension exists: (n, *spatial) → (n, *spatial, 1)
        if self.a.ndim == 3:
            self.a = self.a.unsqueeze(-1)
        if self.u.ndim == 3:
            self.u = self.u.unsqueeze(-1)

        # Resample to target resolution if requested
        if self.resolution is not None:
            self.a = self._resample(self.a, self.resolution)
            self.u = self._resample(self.u, self.resolution)

        # Build coordinate grid
        spatial_dims = self.a.shape[1:-1]  # e.g. (64, 64)
        self.grid = self._make_grid(spatial_dims)

        logger.info(
            "Loaded %s split: a=%s, u=%s",
            self.split, self.a.shape, self.u.shape,
        )

    def _fit_normalizers(self) -> None:
        """Fit normalizers on current data (should be train split)."""
        self.a_normalizer = UnitGaussianNormalizer()
        self.a_normalizer.fit(self.a)
        self.a = self.a_normalizer.encode(self.a)

        self.u_normalizer = UnitGaussianNormalizer()
        self.u_normalizer.fit(self.u)
        self.u = self.u_normalizer.encode(self.u)

    @staticmethod
    def _resample(x: Tensor, resolution: int) -> Tensor:
        """Resample spatial dimensions to target resolution via bilinear interpolation.

        Args:
            x: Field tensor, shape (n, H, W, C).
            resolution: Target spatial resolution.

        Returns:
            Resampled tensor, shape (n, resolution, resolution, C).
        """
        # Move channels to dim 1 for F.interpolate: (n, C, H, W)
        x_perm = x.permute(0, 3, 1, 2)
        x_resampled = F.interpolate(
            x_perm, size=(resolution, resolution), mode="bilinear", align_corners=True,
        )
        # Move back: (n, resolution, resolution, C)
        return x_resampled.permute(0, 2, 3, 1)

    @staticmethod
    def _make_grid(spatial_dims: tuple[int, ...]) -> Tensor:
        """Build a uniform [0,1]^d coordinate grid.

        Args:
            spatial_dims: Spatial dimension sizes.

        Returns:
            Grid tensor, shape (*spatial_dims, d).
        """
        linspaces = [torch.linspace(0, 1, s) for s in spatial_dims]
        meshes = torch.meshgrid(*linspaces, indexing="ij")
        return torch.stack(meshes, dim=-1)

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.a[idx], self.u[idx], self.grid


class InMemoryDataset(Dataset):
    """Simple in-memory dataset from pre-loaded tensors.

    Useful for testing and for data that fits entirely in memory.

    Args:
        a: Input function values, shape (n_samples, *spatial, in_channels).
        u: Target function values, shape (n_samples, *spatial, out_channels).
        normalize: Whether to apply normalization.
    """

    def __init__(
        self,
        a: Tensor,
        u: Tensor,
        normalize: bool = False,
    ) -> None:
        self.a = a
        self.u = u

        # Ensure channel dim
        if self.a.ndim == 3:
            self.a = self.a.unsqueeze(-1)
        if self.u.ndim == 3:
            self.u = self.u.unsqueeze(-1)

        self.a_normalizer: Optional[UnitGaussianNormalizer] = None
        self.u_normalizer: Optional[UnitGaussianNormalizer] = None

        if normalize:
            self.a_normalizer = UnitGaussianNormalizer().fit(self.a)
            self.a = self.a_normalizer.encode(self.a)
            self.u_normalizer = UnitGaussianNormalizer().fit(self.u)
            self.u = self.u_normalizer.encode(self.u)

        # Build coordinate grid
        spatial_dims = self.a.shape[1:-1]
        self.grid = PDEDataset._make_grid(spatial_dims)

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.a[idx], self.u[idx], self.grid

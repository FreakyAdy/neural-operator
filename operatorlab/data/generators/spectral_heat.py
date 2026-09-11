"""Spectral solver for the 2D heat equation.

Generates training data by solving ∂u/∂t = α∇²u on periodic [0,1]²
using exact Fourier-space time evolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_heat_data(
    n_samples: int,
    resolution: int,
    alpha: float = 0.01,
    T: float = 1.0,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Generate heat equation training data.

    Uses spectral methods for exact time evolution of each Fourier mode.

    Args:
        n_samples: Number of IC/solution pairs.
        resolution: Grid resolution per dimension.
        alpha: Thermal diffusivity.
        T: Final time.
        seed: Random seed.
        output_dir: If provided, save to HDF5 in this directory.

    Returns:
        Dict with 'a' (ICs) and 'u' (solutions), each shape (n, res, res, 1).
    """
    from operatorlab.physics.heat import HeatEquation2D

    pde = HeatEquation2D(alpha=alpha, T=T)
    data = pde.generate_dataset(
        n_samples=n_samples,
        resolution=resolution,
        dt=1e-3,  # unused for spectral heat solver
        T=T,
        seed=seed,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"heat_{resolution}_{n_samples}_{seed}.h5"
        filepath = output_dir / filename

        with h5py.File(filepath, "w") as f:
            f.create_dataset("a", data=data["a"].numpy())
            f.create_dataset("u", data=data["u"].numpy())
            f.attrs["alpha"] = alpha
            f.attrs["T"] = T
            f.attrs["resolution"] = resolution
            f.attrs["n_samples"] = n_samples
            f.attrs["seed"] = seed

        logger.info("Saved heat equation data to %s", filepath)

    return data

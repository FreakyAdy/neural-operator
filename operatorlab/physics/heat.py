"""2D Heat Equation PDE problem.

∂u/∂t = α∇²u on [0,1]² with periodic boundary conditions.

The heat equation is the simplest benchmark. Uses spectral methods
for exact (up to numerical precision) data generation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


class HeatEquation2D(PDEProblem):
    """2D Heat equation on periodic [0,1]² domain.

    Generates training data using spectral methods: initial conditions are
    random superpositions of Fourier modes, and the solution is computed
    analytically in Fourier space (each mode decays exponentially).

    Args:
        alpha: Thermal diffusivity coefficient.
        T: Final time for the solution.
    """

    name = "heat"
    spatial_dim = 2

    def __init__(self, alpha: float = 0.01, T: float = 1.0) -> None:
        self.alpha = alpha
        self.T = T

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1e-3,
        T: float | None = None,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate heat equation dataset using spectral solver.

        Initial conditions are random superpositions of low-frequency
        Fourier modes. Solutions are computed analytically in Fourier space.

        Args:
            n_samples: Number of IC/solution pairs.
            resolution: Grid resolution per spatial dimension.
            dt: Not used (solution is exact in Fourier space). Kept for API compat.
            T: Final time. Uses self.T if None.
            seed: Random seed.

        Returns:
            Dict with 'a' (initial conditions) and 'u' (solutions at time T).
        """
        if T is None:
            T = self.T

        rng = np.random.default_rng(seed)
        s = resolution

        # Wavenumbers for [0, 1]² periodic domain
        k_x = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi  # (s,)
        k_y = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi  # (s,)
        kx, ky = np.meshgrid(k_x, k_y, indexing="ij")     # (s, s)
        k_sq = kx**2 + ky**2                                # (s, s)

        # Decay factor: exp(-α|k|²T) for each mode
        decay = np.exp(-self.alpha * k_sq * T)  # (s, s)

        a_all = np.zeros((n_samples, s, s), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s), dtype=np.float64)

        # Physical grid
        x = np.linspace(0, 1, s, endpoint=False)
        y = np.linspace(0, 1, s, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # Number of random Fourier modes to superpose
        n_modes = min(8, s // 2)

        for i in range(n_samples):
            # Random initial condition: superposition of Fourier modes
            a = np.zeros((s, s), dtype=np.float64)
            for _ in range(n_modes):
                kx_rand = rng.integers(-n_modes, n_modes + 1)
                ky_rand = rng.integers(-n_modes, n_modes + 1)
                amp = rng.standard_normal()
                phase = rng.uniform(0, 2 * np.pi)
                a += amp * np.cos(
                    2 * np.pi * (kx_rand * xx + ky_rand * yy) + phase
                )

            # Normalize to reasonable range
            a = a / max(np.abs(a).max(), 1e-8)

            # Solve in Fourier space: u_hat(k, T) = a_hat(k) * exp(-α|k|²T)
            a_hat = np.fft.fft2(a)
            u_hat = a_hat * decay
            u = np.real(np.fft.ifft2(u_hat))

            a_all[i] = a
            u_all[i] = u

        logger.info(
            "Generated %d heat equation samples at resolution %d (α=%.4f, T=%.2f)",
            n_samples, resolution, self.alpha, T,
        )

        return {
            "a": torch.tensor(a_all, dtype=torch.float32).unsqueeze(-1),  # (n, s, s, 1)
            "u": torch.tensor(u_all, dtype=torch.float32).unsqueeze(-1),  # (n, s, s, 1)
        }

    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        """Compute heat equation residual: r = ∂u/∂t - α∇²u.

        Approximates ∂u/∂t as zero (steady-state assumption at eval time)
        and computes the Laplacian via spectral method.

        Args:
            u: Field, shape (batch, H, W, 1).
            t: Time (unused for steady-state residual).

        Returns:
            Residual tensor, shape (batch, H, W, 1).
        """
        # Extract spatial field: (batch, H, W)
        field = u[..., 0]
        batch_size, h, w = field.shape

        # Compute Laplacian via FFT
        k_x = torch.fft.fftfreq(h, d=1.0 / h, device=u.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(w, d=1.0 / w, device=u.device) * 2 * torch.pi
        kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")
        k_sq = kx**2 + ky**2  # (H, W)

        field_hat = torch.fft.fft2(field)               # (batch, H, W)
        laplacian_hat = -k_sq.unsqueeze(0) * field_hat   # (batch, H, W)
        laplacian = torch.fft.ifft2(laplacian_hat).real  # (batch, H, W)

        # Residual: -α∇²u (assuming steady state, ∂u/∂t ≈ 0)
        residual = -self.alpha * laplacian

        return residual.unsqueeze(-1)  # (batch, H, W, 1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        """Check energy conservation for the heat equation.

        For the heat equation, total thermal energy (integral of u) should
        decrease monotonically. We report the mean field value as a proxy.

        Args:
            u: Field, shape (batch, H, W, 1).

        Returns:
            Dict with 'mean_energy' (mean of |u|²).
        """
        energy = (u**2).mean().item()
        return {"mean_energy": energy}

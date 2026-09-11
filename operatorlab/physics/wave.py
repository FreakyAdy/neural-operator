"""2D Wave Equation.

∂²u/∂t² = c²∇²u on periodic [0,1]² domain.

Uses spectral methods with staggered (leapfrog) time stepping.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem


class WaveEquation2D(PDEProblem):
    """2D wave equation on periodic domain.

    Args:
        c: Wave speed.
        T: Final time.
    """

    name = "wave"
    spatial_dim = 2

    def __init__(self, c: float = 1.0, T: float = 1.0) -> None:
        self.c = c
        self.T = T

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1e-3,
        T: float | None = None,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate wave equation data using spectral leapfrog method."""
        if T is None:
            T = self.T

        rng = np.random.default_rng(seed)
        s = resolution
        n_steps = int(T / dt)

        k = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k_sq = kx**2 + ky**2
        omega = self.c * np.sqrt(k_sq + 1e-12)

        x = np.linspace(0, 1, s, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")

        a_all = np.zeros((n_samples, s, s), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s), dtype=np.float64)

        for i in range(n_samples):
            # Random IC: superposition of low-frequency modes
            u0 = np.zeros((s, s))
            for _ in range(6):
                kx_r = rng.integers(-4, 5)
                ky_r = rng.integers(-4, 5)
                amp = rng.standard_normal() / max(abs(kx_r) + abs(ky_r), 1)
                phase = rng.uniform(0, 2 * np.pi)
                u0 += amp * np.sin(2 * np.pi * (kx_r * xx + ky_r * yy) + phase)

            # Exact solution in Fourier space
            u0_hat = np.fft.fft2(u0)
            # u(x,t) = cos(ω*t) * u0_hat (zero initial velocity)
            uT_hat = np.cos(omega * T) * u0_hat
            uT = np.real(np.fft.ifft2(uT_hat))

            a_all[i] = u0
            u_all[i] = uT

        return {
            "a": torch.tensor(a_all, dtype=torch.float32).unsqueeze(-1),
            "u": torch.tensor(u_all, dtype=torch.float32).unsqueeze(-1),
        }

    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        """Compute wave equation residual (Laplacian diagnostic)."""
        field = u[..., 0]
        h, w = field.shape[-2:]
        k_x = torch.fft.fftfreq(h, d=1.0 / h, device=u.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(w, d=1.0 / w, device=u.device) * 2 * torch.pi
        kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")
        k_sq = kx**2 + ky**2
        lap_hat = -k_sq.unsqueeze(0) * torch.fft.fft2(field)
        lap = torch.fft.ifft2(lap_hat).real
        residual = -self.c**2 * lap
        return residual.unsqueeze(-1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        """Total energy should be conserved."""
        energy = (u**2).mean().item()
        return {"energy": energy}

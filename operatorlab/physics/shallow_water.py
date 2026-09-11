"""2D Shallow Water Equations.

∂h/∂t + ∇·(hu) = 0
∂(hu)/∂t + ∇·(hu⊗u) + g*h*∇h = 0

Simplified 1-layer model on periodic [0,1]².
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem


class ShallowWater2D(PDEProblem):
    """2D Shallow water equations.

    Args:
        g: Gravitational acceleration.
        T: Final time.
        H0: Mean water depth.
    """

    name = "shallow_water"
    spatial_dim = 2

    def __init__(self, g: float = 9.81, T: float = 0.1, H0: float = 1.0) -> None:
        self.g = g
        self.T = T
        self.H0 = H0

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1e-4,
        T: float | None = None,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate shallow water data using linearized spectral solver."""
        if T is None:
            T = self.T

        rng = np.random.default_rng(seed)
        s = resolution
        n_steps = int(T / dt)

        k = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k_sq = kx**2 + ky**2
        omega = np.sqrt(self.g * self.H0 * k_sq + 1e-12)

        x = np.linspace(0, 1, s, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")

        a_all = np.zeros((n_samples, s, s), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s), dtype=np.float64)

        for i in range(n_samples):
            # Perturbation in height
            h_pert = np.zeros((s, s))
            for _ in range(4):
                kx_r = rng.integers(-3, 4)
                ky_r = rng.integers(-3, 4)
                amp = 0.01 * rng.standard_normal()
                phase = rng.uniform(0, 2 * np.pi)
                h_pert += amp * np.cos(2 * np.pi * (kx_r * xx + ky_r * yy) + phase)

            # Linearized evolution
            h_hat = np.fft.fft2(h_pert)
            hT_hat = h_hat * np.cos(omega * T)
            hT = np.real(np.fft.ifft2(hT_hat))

            a_all[i] = h_pert
            u_all[i] = hT

        return {
            "a": torch.tensor(a_all, dtype=torch.float32).unsqueeze(-1),
            "u": torch.tensor(u_all, dtype=torch.float32).unsqueeze(-1),
        }

    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        field = u[..., 0]
        h, w = field.shape[-2:]
        k_x = torch.fft.fftfreq(h, d=1.0 / h, device=u.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(w, d=1.0 / w, device=u.device) * 2 * torch.pi
        kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")
        k_sq = kx**2 + ky**2
        lap_hat = -k_sq.unsqueeze(0) * torch.fft.fft2(field)
        lap = torch.fft.ifft2(lap_hat).real
        return (-self.g * self.H0 * lap).unsqueeze(-1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        mass = u.mean().item()
        return {"mass": mass}

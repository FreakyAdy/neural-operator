"""2D Elasticity problem.

Linear elasticity: -∇·σ = f, σ = C:ε, ε = (∇u + ∇uᵀ)/2

Simplified version using scalar Poisson-like problem with
stress/displacement interpretation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem


class Elasticity2D(PDEProblem):
    """2D linear elasticity problem.

    Simplified as a scalar elliptic problem: -∇²u = f on [0,1]²
    with periodic BCs, where f represents body forces.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio.
    """

    name = "elasticity"
    spatial_dim = 2

    def __init__(self, E: float = 1.0, nu: float = 0.3) -> None:
        self.E = E
        self.nu = nu

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1e-3,
        T: float = 1.0,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate elasticity data: f → u where -∇²u = f."""
        rng = np.random.default_rng(seed)
        s = resolution

        k = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k_sq = kx**2 + ky**2
        k_sq_safe = np.where(k_sq == 0, 1.0, k_sq)

        x = np.linspace(0, 1, s, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")

        a_all = np.zeros((n_samples, s, s), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s), dtype=np.float64)

        for i in range(n_samples):
            f = np.zeros((s, s))
            for _ in range(5):
                kx_r = rng.integers(-4, 5)
                ky_r = rng.integers(-4, 5)
                amp = rng.standard_normal()
                phase = rng.uniform(0, 2 * np.pi)
                f += amp * np.sin(2 * np.pi * (kx_r * xx + ky_r * yy) + phase)

            # Solve -∇²u = f in Fourier space: û = f̂ / |k|²
            f_hat = np.fft.fft2(f)
            u_hat = f_hat / k_sq_safe
            u_hat[0, 0] = 0  # zero mean
            u = np.real(np.fft.ifft2(u_hat))

            a_all[i] = f
            u_all[i] = u

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
        return lap.unsqueeze(-1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        return {"mean_displacement": u.mean().item()}

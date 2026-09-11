"""2D Navier-Stokes in vorticity form.

∂ω/∂t + (u·∇)ω = ν∇²ω + f
∇²ψ = -ω
u = ∂ψ/∂y,  v = -∂ψ/∂x

Solved on periodic [0,1]² using pseudo-spectral methods.
This is the primary benchmark for OperatorLab.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


class NavierStokes2D(PDEProblem):
    """2D Navier-Stokes equations in vorticity-stream function form.

    Uses pseudo-spectral methods with Crank-Nicolson for diffusion
    and Adams-Bashforth for advection on periodic [0,1]².

    Args:
        viscosity: Kinematic viscosity ν. 1e-3 for turbulent, 1e-4 for harder.
        T: Final time.
        forcing_amp: Amplitude of the forcing function.
    """

    name = "navier_stokes"
    spatial_dim = 2

    def __init__(
        self,
        viscosity: float = 1e-3,
        T: float = 1.0,
        forcing_amp: float = 0.1,
    ) -> None:
        self.viscosity = viscosity
        self.T = T
        self.forcing_amp = forcing_amp

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1e-3,
        T: float | None = None,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate Navier-Stokes training data using pseudo-spectral solver.

        Args:
            n_samples: Number of IC/solution pairs.
            resolution: Spatial grid resolution.
            dt: Time step for the solver.
            T: Final time. Uses self.T if None.
            seed: Random seed.

        Returns:
            Dict with 'a' (initial vorticity) and 'u' (vorticity at T).
        """
        if T is None:
            T = self.T

        from operatorlab.data.generators.spectral_ns import solve_navier_stokes_2d

        rng = np.random.default_rng(seed)
        s = resolution

        a_all = np.zeros((n_samples, s, s), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s), dtype=np.float64)

        for i in range(n_samples):
            # Generate random initial vorticity as superposition of Fourier modes
            omega_0 = self._random_vorticity(s, rng)
            omega_T = solve_navier_stokes_2d(
                omega_0, s, dt, T,
                viscosity=self.viscosity,
                forcing_amp=self.forcing_amp,
            )
            a_all[i] = omega_0
            u_all[i] = omega_T

            if (i + 1) % max(1, n_samples // 10) == 0:
                logger.info("Generated %d/%d NS samples", i + 1, n_samples)

        return {
            "a": torch.tensor(a_all, dtype=torch.float32).unsqueeze(-1),
            "u": torch.tensor(u_all, dtype=torch.float32).unsqueeze(-1),
        }

    @staticmethod
    def _random_vorticity(resolution: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a random vorticity field as a superposition of Fourier modes.

        Args:
            resolution: Grid resolution.
            rng: NumPy random number generator.

        Returns:
            Vorticity field, shape (resolution, resolution).
        """
        s = resolution
        x = np.linspace(0, 1, s, endpoint=False)
        y = np.linspace(0, 1, s, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        omega = np.zeros((s, s), dtype=np.float64)
        n_modes = 6
        for _ in range(n_modes):
            kx = rng.integers(-4, 5)
            ky = rng.integers(-4, 5)
            if kx == 0 and ky == 0:
                continue
            amp = rng.standard_normal() / max(abs(kx) + abs(ky), 1)
            phase = rng.uniform(0, 2 * np.pi)
            omega += amp * np.sin(2 * np.pi * (kx * xx + ky * yy) + phase)

        return omega

    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        """Compute NS residual: r = ν∇²ω + f - (u·∇)ω.

        Simplified diagnostic: computes ν∇²ω + f (ignoring the advection
        term which requires velocity reconstruction).

        Args:
            u: Vorticity field, shape (batch, H, W, 1).
            t: Time.

        Returns:
            Residual tensor, shape (batch, H, W, 1).
        """
        omega = u[..., 0]  # (batch, H, W)
        batch_size, h, w = omega.shape

        k_x = torch.fft.fftfreq(h, d=1.0 / h, device=u.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(w, d=1.0 / w, device=u.device) * 2 * torch.pi
        kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")
        k_sq = kx**2 + ky**2

        omega_hat = torch.fft.fft2(omega)
        laplacian_hat = -k_sq.unsqueeze(0) * omega_hat
        laplacian = torch.fft.ifft2(laplacian_hat).real

        # Forcing
        x = torch.linspace(0, 1, h, device=u.device)
        y = torch.linspace(0, 1, w, device=u.device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        forcing = self.forcing_amp * (
            torch.sin(2 * torch.pi * (xx + yy))
            + torch.cos(2 * torch.pi * (xx + yy))
        )

        residual = self.viscosity * laplacian + forcing.unsqueeze(0)
        return residual.unsqueeze(-1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        """Check enstrophy (integral of ω²/2) as a diagnostic.

        Args:
            u: Vorticity field, shape (batch, H, W, 1).

        Returns:
            Dict with 'enstrophy' and 'mean_vorticity'.
        """
        omega = u[..., 0]
        enstrophy = 0.5 * (omega**2).mean().item()
        mean_vort = omega.mean().item()
        return {
            "enstrophy": enstrophy,
            "mean_vorticity": mean_vort,
        }

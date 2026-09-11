"""2D Reaction-Diffusion System (Gray-Scott model).

∂u/∂t = d_u∇²u - u*v² + F*(1-u)
∂v/∂t = d_v∇²v + u*v² - (F+k)*v

Two-component field. Tests multi-channel operator learning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from operatorlab.physics.base import PDEProblem


class ReactionDiffusion2D(PDEProblem):
    """2D Gray-Scott reaction-diffusion system.

    Args:
        d_u: Diffusion coefficient for u.
        d_v: Diffusion coefficient for v.
        F: Feed rate.
        k: Kill rate.
        T: Final time.
    """

    name = "reaction_diffusion"
    spatial_dim = 2

    def __init__(
        self,
        d_u: float = 2e-5,
        d_v: float = 1e-5,
        F: float = 0.04,
        k: float = 0.06,
        T: float = 1.0,
    ) -> None:
        self.d_u = d_u
        self.d_v = d_v
        self.F = F
        self.k = k
        self.T = T

    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float = 1.0,
        T: float | None = None,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate Gray-Scott data using semi-implicit spectral method."""
        if T is None:
            T = self.T

        rng = np.random.default_rng(seed)
        s = resolution
        n_steps = int(T / dt)

        k_arr = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi
        kx, ky = np.meshgrid(k_arr, k_arr, indexing="ij")
        k_sq = kx**2 + ky**2

        # Implicit diffusion operators
        diff_u = 1.0 / (1.0 + self.d_u * dt * k_sq)
        diff_v = 1.0 / (1.0 + self.d_v * dt * k_sq)

        a_all = np.zeros((n_samples, s, s, 2), dtype=np.float64)
        u_all = np.zeros((n_samples, s, s, 2), dtype=np.float64)

        for i in range(n_samples):
            # Initial condition: u=1, v=0 with random perturbation patches
            u = np.ones((s, s))
            v = np.zeros((s, s))

            # Add random square patches of v=0.5, u=0.5
            n_patches = rng.integers(1, 5)
            for _ in range(n_patches):
                cx = rng.integers(s // 4, 3 * s // 4)
                cy = rng.integers(s // 4, 3 * s // 4)
                pw = rng.integers(2, max(3, s // 8))
                x_lo = max(0, cx - pw)
                x_hi = min(s, cx + pw)
                y_lo = max(0, cy - pw)
                y_hi = min(s, cy + pw)
                u[x_lo:x_hi, y_lo:y_hi] = 0.5
                v[x_lo:x_hi, y_lo:y_hi] = 0.25
                # Add noise
                u[x_lo:x_hi, y_lo:y_hi] += 0.01 * rng.standard_normal((x_hi-x_lo, y_hi-y_lo))
                v[x_lo:x_hi, y_lo:y_hi] += 0.01 * rng.standard_normal((x_hi-x_lo, y_hi-y_lo))

            a_all[i, :, :, 0] = u
            a_all[i, :, :, 1] = v

            # Time integration
            for _ in range(n_steps):
                uvv = u * v * v
                # Reaction
                u_react = -uvv + self.F * (1.0 - u)
                v_react = uvv - (self.F + self.k) * v

                # Semi-implicit: diffusion in Fourier, reaction explicit
                u_hat = np.fft.fft2(u + dt * u_react)
                v_hat = np.fft.fft2(v + dt * v_react)
                u = np.real(np.fft.ifft2(u_hat * diff_u))
                v = np.real(np.fft.ifft2(v_hat * diff_v))

                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)

            u_all[i, :, :, 0] = u
            u_all[i, :, :, 1] = v

        return {
            "a": torch.tensor(a_all, dtype=torch.float32),  # (n, s, s, 2)
            "u": torch.tensor(u_all, dtype=torch.float32),  # (n, s, s, 2)
        }

    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        u_field = u[..., 0]
        v_field = u[..., 1]
        h, w = u_field.shape[-2:]

        k_x = torch.fft.fftfreq(h, d=1.0 / h, device=u.device) * 2 * torch.pi
        k_y = torch.fft.fftfreq(w, d=1.0 / w, device=u.device) * 2 * torch.pi
        kx, ky = torch.meshgrid(k_x, k_y, indexing="ij")
        k_sq = kx**2 + ky**2

        lap_u = torch.fft.ifft2(-k_sq.unsqueeze(0) * torch.fft.fft2(u_field)).real
        lap_v = torch.fft.ifft2(-k_sq.unsqueeze(0) * torch.fft.fft2(v_field)).real

        uvv = u_field * v_field * v_field
        res_u = self.d_u * lap_u - uvv + self.F * (1 - u_field)
        res_v = self.d_v * lap_v + uvv - (self.F + self.k) * v_field

        return torch.stack([res_u, res_v], dim=-1)

    def check_conservation(self, u: Tensor) -> dict[str, float]:
        return {
            "total_u": u[..., 0].mean().item(),
            "total_v": u[..., 1].mean().item(),
        }

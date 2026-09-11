"""Pseudo-spectral solver for 2D Navier-Stokes in vorticity form.

∂ω/∂t + (u·∇)ω = ν∇²ω + f
∇²ψ = -ω
u = ∂ψ/∂y,  v = -∂ψ/∂x

Uses Crank-Nicolson for diffusion and Adams-Bashforth (2nd order)
for advection. 2/3 dealiasing rule applied to the nonlinear term.
"""

from __future__ import annotations

import numpy as np


def solve_navier_stokes_2d(
    omega_0: np.ndarray,
    resolution: int,
    dt: float,
    T: float,
    viscosity: float = 1e-3,
    forcing_amp: float = 0.1,
) -> np.ndarray:
    """Solve 2D Navier-Stokes on periodic [0,1]² using pseudo-spectral method.

    Args:
        omega_0: Initial vorticity field, shape (resolution, resolution).
        resolution: Spatial grid resolution.
        dt: Time step.
        T: Final time.
        viscosity: Kinematic viscosity ν.
        forcing_amp: Amplitude of the forcing function.

    Returns:
        Vorticity field at time T, shape (resolution, resolution).
    """
    s = resolution
    n_steps = int(T / dt)

    # Wavenumbers for periodic [0, 1]² domain
    k = np.fft.fftfreq(s, d=1.0 / s) * 2 * np.pi  # (s,)
    kx, ky = np.meshgrid(k, k, indexing="ij")  # (s, s)
    k_sq = kx**2 + ky**2  # (s, s)

    # Avoid division by zero at k=(0,0)
    k_sq_safe = np.where(k_sq == 0, 1.0, k_sq)

    # Dealiasing mask: zero out top 1/3 of modes (2/3 rule)
    dealias_mask = np.ones((s, s), dtype=np.float64)
    kmax = s // 3
    for i in range(s):
        for j in range(s):
            if abs(k[i] / (2 * np.pi)) > kmax or abs(k[j] / (2 * np.pi)) > kmax:
                dealias_mask[i, j] = 0.0

    # Crank-Nicolson coefficients for diffusion
    # (1 + 0.5*ν*dt*|k|²) * ω̂_{n+1} = (1 - 0.5*ν*dt*|k|²) * ω̂_n + dt * NL
    cn_denom = 1.0 + 0.5 * viscosity * dt * k_sq  # (s, s)
    cn_numer = 1.0 - 0.5 * viscosity * dt * k_sq  # (s, s)

    # Forcing function in physical space: f(x,y) = A*(sin(2π(x+y)) + cos(2π(x+y)))
    x = np.linspace(0, 1, s, endpoint=False)
    y = np.linspace(0, 1, s, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    forcing = forcing_amp * (
        np.sin(2 * np.pi * (xx + yy)) + np.cos(2 * np.pi * (xx + yy))
    )
    forcing_hat = np.fft.fft2(forcing)

    # Initialize
    omega_hat = np.fft.fft2(omega_0)

    # For Adams-Bashforth: store previous nonlinear term
    nl_prev = None

    for step in range(n_steps):
        # Compute velocity from vorticity via stream function
        # ψ̂ = -ω̂ / |k|²
        psi_hat = -omega_hat / k_sq_safe
        psi_hat[0, 0] = 0  # zero mean stream function

        # u = ∂ψ/∂y = i*ky*ψ̂,  v = -∂ψ/∂x = -i*kx*ψ̂
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat

        # Compute ∂ω/∂x and ∂ω/∂y in Fourier space
        domega_dx_hat = 1j * kx * omega_hat
        domega_dy_hat = 1j * ky * omega_hat

        # Transform to physical space with dealiasing
        u = np.real(np.fft.ifft2(u_hat * dealias_mask))
        v = np.real(np.fft.ifft2(v_hat * dealias_mask))
        domega_dx = np.real(np.fft.ifft2(domega_dx_hat * dealias_mask))
        domega_dy = np.real(np.fft.ifft2(domega_dy_hat * dealias_mask))

        # Nonlinear term: -(u·∇)ω = -(u*∂ω/∂x + v*∂ω/∂y)
        nl = -(u * domega_dx + v * domega_dy)
        nl_hat = np.fft.fft2(nl) * dealias_mask

        # Time stepping: Crank-Nicolson for diffusion, Adams-Bashforth for advection
        if nl_prev is None:
            # First step: forward Euler for advection
            rhs = cn_numer * omega_hat + dt * (nl_hat + forcing_hat)
        else:
            # AB2 for advection
            rhs = cn_numer * omega_hat + dt * (
                1.5 * nl_hat - 0.5 * nl_prev + forcing_hat
            )

        omega_hat = rhs / cn_denom

        nl_prev = nl_hat.copy()

    # Transform back to physical space
    omega_T = np.real(np.fft.ifft2(omega_hat))

    return omega_T

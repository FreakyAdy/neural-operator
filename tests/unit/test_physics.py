"""Unit tests for PDE systems and data generators."""

from __future__ import annotations

import pytest
import torch

from operatorlab.physics.elasticity import Elasticity2D
from operatorlab.physics.heat import HeatEquation2D
from operatorlab.physics.navier_stokes import NavierStokes2D
from operatorlab.physics.reaction_diffusion import ReactionDiffusion2D
from operatorlab.physics.shallow_water import ShallowWater2D
from operatorlab.physics.wave import WaveEquation2D


class TestPDEProblems:
    def test_heat_equation(self) -> None:
        pde = HeatEquation2D(alpha=0.01, T=0.5)
        data = pde.generate_dataset(n_samples=4, resolution=16, seed=1)
        assert data["a"].shape == (4, 16, 16, 1)
        assert data["u"].shape == (4, 16, 16, 1)
        res = pde.compute_residual(data["u"], t=0.5)
        assert res.shape == (4, 16, 16, 1)
        cons = pde.check_conservation(data["u"])
        assert "mean_energy" in cons

    def test_navier_stokes(self) -> None:
        pde = NavierStokes2D(viscosity=1e-3, T=0.1)
        data = pde.generate_dataset(n_samples=2, resolution=16, seed=1)
        assert data["a"].shape == (2, 16, 16, 1)
        assert data["u"].shape == (2, 16, 16, 1)
        cons = pde.check_conservation(data["u"])
        assert "enstrophy" in cons
        assert "mean_vorticity" in cons

    def test_wave_equation(self) -> None:
        pde = WaveEquation2D(c=1.0, T=0.5)
        data = pde.generate_dataset(n_samples=3, resolution=16, seed=1)
        assert data["a"].shape == (3, 16, 16, 1)
        assert data["u"].shape == (3, 16, 16, 1)
        cons = pde.check_conservation(data["u"])
        assert "energy" in cons

    def test_shallow_water(self) -> None:
        pde = ShallowWater2D(T=0.2)
        data = pde.generate_dataset(n_samples=2, resolution=16, seed=1)
        assert data["a"].shape == (2, 16, 16, 1)
        assert data["u"].shape == (2, 16, 16, 1)
        cons = pde.check_conservation(data["u"])
        assert "mass" in cons

    def test_elasticity(self) -> None:
        pde = Elasticity2D()
        data = pde.generate_dataset(n_samples=2, resolution=16, seed=1)
        assert data["a"].shape == (2, 16, 16, 1)
        assert data["u"].shape == (2, 16, 16, 1)
        cons = pde.check_conservation(data["u"])
        assert "mean_displacement" in cons

    def test_reaction_diffusion(self) -> None:
        pde = ReactionDiffusion2D(T=0.5)
        data = pde.generate_dataset(n_samples=2, resolution=16, seed=1)
        assert data["a"].shape == (2, 16, 16, 2)
        assert data["u"].shape == (2, 16, 16, 2)
        cons = pde.check_conservation(data["u"])
        assert "total_u" in cons
        assert "total_v" in cons

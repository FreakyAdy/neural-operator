"""Abstract base class for PDE problem definitions.

Every PDE problem in OperatorLab must subclass PDEProblem and implement
the methods for data generation, residual computation, and conservation checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class PDEProblem(ABC):
    """Base class for PDE problems.

    Defines the interface for:
        - Generating training/test datasets via numerical solvers
        - Computing PDE residuals for physics-informed training
        - Checking conservation law violations

    Attributes:
        name: Short identifier for this PDE (e.g. 'heat', 'navier_stokes').
        spatial_dim: Number of spatial dimensions (1, 2, or 3).
    """

    name: str
    spatial_dim: int

    @abstractmethod
    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float,
        T: float,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """Generate training data by solving the PDE numerically.

        Args:
            n_samples: Number of initial condition / solution pairs.
            resolution: Spatial grid resolution per dimension.
            dt: Time step for the numerical solver.
            T: Final time.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys:
                'a': Initial condition, shape (n_samples, *spatial_grid, channels).
                'u': Target field at time T, shape (n_samples, *spatial_grid, channels).
        """

    @abstractmethod
    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        """Compute the PDE residual of a predicted field.

        Used as a physics-informed diagnostic. Lower residual means
        the prediction better satisfies the PDE.

        Args:
            u: Predicted field, shape (batch, *spatial, channels).
            t: Time at which to evaluate the residual.

        Returns:
            Residual tensor, shape (batch, *spatial, channels).
        """

    @abstractmethod
    def check_conservation(self, u: Tensor) -> dict[str, float]:
        """Check violations of conserved quantities.

        Args:
            u: Predicted field, shape (batch, *spatial, channels).

        Returns:
            Dict mapping quantity names to violation magnitudes.
            Example: {'mass': 1e-4, 'energy': 2e-3}
        """

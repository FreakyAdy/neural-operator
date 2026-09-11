"""Input noise and sensor perturbation stress testing.

Evaluates operator sensitivity to measurement noise, sensor dropouts, and impulsive errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.evaluation.stress import perturb_gaussian_noise, perturb_sensor_sparsity, perturb_shot_noise
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class NoiseStressResult:
    """Outcome of input noise stress test."""

    noise_type: str
    noise_level: str
    clean_l2: float
    noisy_l2: float
    degradation_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def evaluate_noise_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    resolution: int = 64,
    sigma: float = 0.05,
    n_samples: int = 20,
    device: str = "cpu",
) -> NoiseStressResult:
    """Stress test operator robustness against input Gaussian noise."""
    model.eval()

    base_data = pde.generate_dataset(n_samples=n_samples, resolution=resolution, seed=501)
    a = base_data["a"].to(device)
    u = base_data["u"].to(device)

    with torch.no_grad():
        pred_clean = model(a)
    clean_l2 = relative_l2_error(pred_clean, u).item()

    # Add Gaussian noise
    a_noisy = perturb_gaussian_noise(a, sigma=sigma, relative=True)

    with torch.no_grad():
        pred_noisy = model(a_noisy)
    noisy_l2 = relative_l2_error(pred_noisy, u).item()

    degradation = noisy_l2 / max(clean_l2, 1e-8)

    if degradation < 1.8:
        status = "PASS"
    elif degradation < 3.5:
        status = "WARN"
    else:
        status = "FAIL"

    return NoiseStressResult(
        noise_type="Input Noise",
        noise_level=f"σ = {sigma:.2f}",
        clean_l2=clean_l2,
        noisy_l2=noisy_l2,
        degradation_ratio=degradation,
        status=status,
        message=f"Input Noise σ={sigma:.2f} ({degradation:.2f}x)",
    )

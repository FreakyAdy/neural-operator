"""Parameter OOD stress testing.

Evaluates how far neural operators generalize when physical coefficients
(viscosity, diffusivity, wave speed, reaction constants) drift outside the training distribution.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class ParameterStressResult:
    """Outcome of physical parameter shift stress test."""

    parameter_name: str
    base_val: float
    shifted_val: float
    base_l2: float
    shifted_l2: float
    degradation_ratio: float
    status: str  # PASS, WARN, FAIL
    message: str


def evaluate_parameter_stress(
    model: NeuralOperator,
    pde: PDEProblem,
    resolution: int = 64,
    shift_multiplier: float = 2.5,
    n_samples: int = 20,
    device: str = "cpu",
) -> ParameterStressResult:
    """Stress test physical parameter extrapolation."""
    model.eval()

    # Identify relevant parameter
    shifted_pde = copy.deepcopy(pde)
    param_name = "param"
    base_val = 1.0
    shifted_val = 2.5

    if hasattr(shifted_pde, "viscosity"):
        param_name = "ν"
        base_val = pde.viscosity
        shifted_pde.viscosity *= shift_multiplier
        shifted_val = shifted_pde.viscosity
    elif hasattr(shifted_pde, "alpha"):
        param_name = "α"
        base_val = pde.alpha
        shifted_pde.alpha *= shift_multiplier
        shifted_val = shifted_pde.alpha
    elif hasattr(shifted_pde, "diffusivity"):
        param_name = "α"
        base_val = pde.diffusivity
        shifted_pde.diffusivity *= shift_multiplier
        shifted_val = shifted_pde.diffusivity
    elif hasattr(shifted_pde, "c"):
        param_name = "c"
        base_val = pde.c
        shifted_pde.c *= shift_multiplier
        shifted_val = shifted_pde.c

    # Baseline data
    base_data = pde.generate_dataset(n_samples=n_samples, resolution=resolution, seed=201)
    a_base = base_data["a"].to(device)
    u_base = base_data["u"].to(device)
    with torch.no_grad():
        base_l2 = relative_l2_error(model(a_base), u_base).item()

    # Shifted data
    shifted_data = shifted_pde.generate_dataset(n_samples=n_samples, resolution=resolution, seed=202)
    a_shift = shifted_data["a"].to(device)
    u_shift = shifted_data["u"].to(device)
    with torch.no_grad():
        shifted_l2 = relative_l2_error(model(a_shift), u_shift).item()

    degradation = shifted_l2 / max(base_l2, 1e-8)

    if degradation < 1.6:
        status = "PASS"
    elif degradation < 3.0:
        status = "WARN"
    else:
        status = "FAIL"

    return ParameterStressResult(
        parameter_name=param_name,
        base_val=base_val,
        shifted_val=shifted_val,
        base_l2=base_l2,
        shifted_l2=shifted_l2,
        degradation_ratio=degradation,
        status=status,
        message=f"{param_name}: {base_val:.2e} → {shifted_val:.2e} ({degradation:.2f}x)",
    )

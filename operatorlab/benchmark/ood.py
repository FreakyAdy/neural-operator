"""Out-of-Distribution (OOD) benchmark harness.

Runs the comprehensive multi-dimensional generalization audit.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from operatorlab.benchmark.reports import GeneralizationAuditReport
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem
from operatorlab.stress.boundary_shift import evaluate_boundary_stress
from operatorlab.stress.geometry_shift import evaluate_geometry_stress
from operatorlab.stress.long_horizon import evaluate_long_horizon_stress
from operatorlab.stress.noise import evaluate_noise_stress
from operatorlab.stress.parameter_shift import evaluate_parameter_stress
from operatorlab.stress.resolution import evaluate_resolution_stress
from operatorlab.validity.conservation import audit_conservation

logger = logging.getLogger(__name__)


def run_generalization_audit(
    model: NeuralOperator,
    pde: PDEProblem,
    base_res: int = 64,
    target_res: int = 256,
    n_samples: int = 20,
    device: str = "cpu",
) -> GeneralizationAuditReport:
    """Execute the full suite of OOD, robustness, and physical conservation stress tests.

    Returns:
        GeneralizationAuditReport matching the OPERATOR GENERALIZATION AUDIT standard.
    """
    model_name = type(model).__name__
    pde_name = getattr(pde, "name", type(pde).__name__)

    logger.info("Executing Generalization Audit on %s (%s)...", model_name, pde_name)

    # 1. Resolution OOD
    res_res = evaluate_resolution_stress(model, pde, base_res=base_res, target_res=target_res, n_samples=n_samples, device=device)

    # 2. Parameter OOD
    param_res = evaluate_parameter_stress(model, pde, resolution=base_res, shift_multiplier=2.5, n_samples=n_samples, device=device)

    # 3. Boundary OOD
    bound_res = evaluate_boundary_stress(model, pde, resolution=base_res, n_samples=n_samples, device=device)

    # 4. Geometry OOD
    geom_res = evaluate_geometry_stress(model, pde, resolution=base_res, distortion_strength=0.08, n_samples=n_samples, device=device)

    # 5. Input Noise
    noise_res = evaluate_noise_stress(model, pde, resolution=base_res, sigma=0.05, n_samples=n_samples, device=device)

    # 6. Long-Horizon Rollout
    rollout_res = evaluate_long_horizon_stress(model, pde, resolution=base_res, target_steps=20, device=device)

    # 7. Physics Conservation
    test_data = pde.generate_dataset(n_samples=n_samples, resolution=base_res, seed=777)
    a = test_data["a"].to(device)
    u = test_data["u"].to(device)
    with torch.no_grad():
        pred = model(a)
    cons_res = audit_conservation(pred, a, u)

    # Compute overall scientific reliability score (0-100)
    score_map = {"PASS": 15, "WARN": 8, "FAIL": 0}
    test_points = (
        score_map[res_res.status]
        + score_map[param_res.status]
        + score_map[bound_res.status]
        + score_map[geom_res.status]
        + score_map[noise_res.status]
        + score_map[rollout_res.status]
    )  # Max 90 points

    # Conservation points (max 10 points)
    cons_points = 0
    if cons_res.mass_error_percent < 1.0:
        cons_points += 5
    elif cons_res.mass_error_percent < 5.0:
        cons_points += 2

    if cons_res.energy_drift_percent < 2.5:
        cons_points += 5
    elif cons_res.energy_drift_percent < 10.0:
        cons_points += 2

    reliability_score = min(100, max(10, test_points + cons_points))

    def _fmt_p(v: float) -> str:
        if v == 0:
            return "0.0"
        if abs(v) < 0.01:
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{v:.2f}"

    return GeneralizationAuditReport(
        model_name=model_name,
        pde_name=pde_name,
        base_res=base_res,
        target_res=target_res,
        resolution_status=res_res.status,
        parameter_desc=f"{param_res.parameter_name}: {_fmt_p(param_res.base_val)} → {_fmt_p(param_res.shifted_val)}",
        parameter_status=param_res.status,
        boundary_desc="Periodic → Dirichlet",
        boundary_status=bound_res.status,
        geometry_desc="Square → irregular domain",
        geometry_status=geom_res.status,
        noise_desc="σ = 0.05",
        noise_status=noise_res.status,
        rollout_desc="t = 1 → 20",
        rollout_status=rollout_res.status,
        mass_error_pct=cons_res.mass_error_percent,
        energy_drift_pct=cons_res.energy_drift_percent,
        reliability_score=reliability_score,
    )

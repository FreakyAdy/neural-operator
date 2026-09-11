"""Out-of-Distribution (OOD) operator generalization suite.

Measures how far neural operators generalize outside their training distribution
across multiple orthogonal shift dimensions:
1. Resolution OOD (zero-shot discretization transfer)
2. Parameter OOD (shifted physical coefficients, e.g. viscosity, wave speed)
3. Physics OOD (source terms, forcing amplitude, initial condition distribution)
4. Geometry OOD (coordinate distortion, non-uniform spatial grids)
5. Combined OOD (simultaneous shifts across all dimensions)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor

from operatorlab.evaluation.metrics import h1_error, relative_l2_error
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


class OODShiftType(str, Enum):
    """Categories of out-of-distribution shifts."""

    IN_DOMAIN = "in_domain"
    RESOLUTION = "resolution"
    PARAMETER = "parameter"
    PHYSICS = "physics"
    GEOMETRY = "geometry"
    COMBINED = "combined"


@dataclass
class OODCaseResult:
    """Evaluation result for a single OOD test condition."""

    axis: str
    name: str
    description: str
    in_domain_l2: float
    ood_l2: float
    degradation_ratio: float
    ood_h1: float = 0.0
    physics_residual: float = 0.0


@dataclass
class OODGeneralizationReport:
    """Comprehensive OOD Generalization report across all shift dimensions."""

    model_name: str
    pde_name: str
    base_resolution: int
    cases: list[OODCaseResult] = field(default_factory=list)

    @property
    def mean_degradation_ratio(self) -> float:
        """Average ratio of OOD error to in-domain error (lower is better, 1.0 = perfect)."""
        ood_cases = [c for c in self.cases if c.axis != OODShiftType.IN_DOMAIN.value]
        if not ood_cases:
            return 1.0
        return sum(c.degradation_ratio for c in ood_cases) / len(ood_cases)

    @property
    def generalization_score(self) -> float:
        """Composite generalization index (0 to 100, 100 = perfect invariance)."""
        # Score penalizes degradation ratio > 1.0 exponentially: 100 * exp(-0.5 * max(0, ratio - 1))
        deg = max(1.0, self.mean_degradation_ratio)
        score = 100.0 * (1.0 / (1.0 + 0.5 * (deg - 1.0)))
        return round(score, 2)

    def summary_table(self) -> str:
        """Format an executive text table of OOD generalization performance."""
        lines = [
            f"=== OOD OPERATOR GENERALIZATION REPORT: {self.model_name} on {self.pde_name} ===",
            f"Base Resolution: {self.base_resolution}×{self.base_resolution} | Generalization Index: {self.generalization_score}/100",
            "-" * 86,
            f"{'Shift Axis':<14} | {'Condition':<20} | {'In-Domain L2':<12} | {'OOD L2':<10} | {'Degradation':<11} | {'OOD H1':<10}",
            "-" * 86,
        ]
        for c in self.cases:
            deg_str = f"{c.degradation_ratio:.2f}x" if c.axis != OODShiftType.IN_DOMAIN.value else "1.00x (ref)"
            lines.append(
                f"{c.axis:<14} | {c.name:<20} | {c.in_domain_l2:<12.6f} | {c.ood_l2:<10.6f} | {deg_str:<11} | {c.ood_h1:<10.6f}"
            )
        lines.append("-" * 86)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON export."""
        return {
            "model_name": self.model_name,
            "pde_name": self.pde_name,
            "base_resolution": self.base_resolution,
            "mean_degradation_ratio": self.mean_degradation_ratio,
            "generalization_score": self.generalization_score,
            "cases": [asdict(c) for c in self.cases],
        }


def warp_grid_coordinates(
    resolution: int,
    distortion_strength: float = 0.05,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a non-uniform / warped coordinate grid in [0, 1]².

    Simulates geometry OOD where spatial sampling is smooth but non-Cartesian.

    Args:
        resolution: Grid resolution N.
        distortion_strength: Amplitude of coordinate perturbation.
        device: Torch device.

    Returns:
        Tensor of shape (1, N, N, 2) with distorted coordinates.
    """
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # Add smooth sinusoidal deformation
    dx = distortion_strength * torch.sin(2 * torch.pi * grid_y)
    dy = distortion_strength * torch.sin(2 * torch.pi * grid_x)

    warped_x = torch.clamp(grid_x + dx, 0.0, 1.0)
    warped_y = torch.clamp(grid_y + dy, 0.0, 1.0)

    return torch.stack([warped_x, warped_y], dim=-1).unsqueeze(0)


@torch.no_grad()
def evaluate_on_dataset(
    model: NeuralOperator,
    dataset: dict[str, Tensor],
    grid: Optional[Tensor] = None,
    batch_size: int = 20,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Evaluate model on a dataset dictionary returning (l2_error, h1_error)."""
    model.eval()
    a_all = dataset["a"].to(device)
    u_all = dataset["u"].to(device)
    n = len(a_all)

    all_preds = []
    for i in range(0, n, batch_size):
        a_b = a_all[i : i + batch_size]
        grid_b = None
        if grid is not None:
            grid_b = grid.to(device).expand(len(a_b), -1, -1, -1)
        pred_b = model(a_b, grid=grid_b)
        all_preds.append(pred_b)

    pred = torch.cat(all_preds, dim=0)
    l2 = relative_l2_error(pred, u_all).item()

    res = u_all.shape[1]
    h1 = h1_error(pred, u_all, dx=1.0 / res).item()
    return l2, h1


def run_ood_generalization_benchmark(
    model: NeuralOperator,
    pde: PDEProblem,
    base_resolution: int = 64,
    n_samples: int = 50,
    device: str = "cpu",
    target_resolutions: Optional[list[int]] = None,
    parameter_multipliers: Optional[list[float]] = None,
    forcing_multipliers: Optional[list[float]] = None,
) -> OODGeneralizationReport:
    """Run the comprehensive OOD operator generalization benchmark.

    Tests:
    1. In-Domain Reference: Baseline resolution and physical parameters.
    2. Resolution OOD: Extrapolation to higher / different grid sizes.
    3. Parameter OOD: Viscosity / diffusion / wave speed shifts outside training range.
    4. Physics OOD: Forcing amplitude / source term shift.
    5. Geometry OOD: Non-Cartesian warped coordinate grid.
    6. Combined OOD: Simultaneous resolution, parameter, and geometry shift.

    Args:
        model: Trained neural operator.
        pde: Base PDE problem instance.
        base_resolution: Training resolution.
        n_samples: Number of test samples per test condition.
        device: Torch device ('cpu' or 'cuda').
        target_resolutions: List of OOD resolutions (defaults to [2*base_res]).
        parameter_multipliers: Multipliers for main PDE coefficient (defaults to [2.0]).
        forcing_multipliers: Multipliers for forcing amplitude (defaults to [2.5]).

    Returns:
        OODGeneralizationReport with full metric breakdown and degradation ratios.
    """
    model_name = type(model).__name__
    pde_name = getattr(pde, "name", type(pde).__name__)
    report = OODGeneralizationReport(
        model_name=model_name,
        pde_name=pde_name,
        base_resolution=base_resolution,
    )

    if target_resolutions is None:
        target_resolutions = [base_resolution * 2]
    if parameter_multipliers is None:
        parameter_multipliers = [2.0]
    if forcing_multipliers is None:
        forcing_multipliers = [2.5]

    logger.info("Generating in-domain baseline test set (%d×%d)...", base_resolution, base_resolution)
    base_data = pde.generate_dataset(n_samples=n_samples, resolution=base_resolution, seed=1001)
    base_l2, base_h1 = evaluate_on_dataset(model, base_data, device=device)

    # Reference in-domain
    report.cases.append(
        OODCaseResult(
            axis=OODShiftType.IN_DOMAIN.value,
            name="Baseline",
            description=f"Standard test distribution ({base_resolution}×{base_resolution})",
            in_domain_l2=base_l2,
            ood_l2=base_l2,
            degradation_ratio=1.0,
            ood_h1=base_h1,
        )
    )

    # 1. Resolution OOD
    for res in target_resolutions:
        logger.info("Running Resolution OOD test at %d×%d...", res, res)
        res_data = pde.generate_dataset(n_samples=n_samples, resolution=res, seed=1002)
        res_l2, res_h1 = evaluate_on_dataset(model, res_data, device=device)
        deg = res_l2 / max(base_l2, 1e-8)
        report.cases.append(
            OODCaseResult(
                axis=OODShiftType.RESOLUTION.value,
                name=f"Res {res}×{res}",
                description=f"Zero-shot resolution scale {res / base_resolution:.1f}x",
                in_domain_l2=base_l2,
                ood_l2=res_l2,
                degradation_ratio=deg,
                ood_h1=res_h1,
            )
        )

    # 2. Parameter OOD (shifted physical coefficients)
    for mult in parameter_multipliers:
        logger.info("Running Parameter OOD test (param mult=%.2f)...", mult)
        shifted_pde = copy.deepcopy(pde)
        # Shift relevant physical parameter depending on PDE type
        param_desc = "shifted parameter"
        if hasattr(shifted_pde, "viscosity"):
            shifted_pde.viscosity *= mult
            param_desc = f"ν={shifted_pde.viscosity:.2e}"
        elif hasattr(shifted_pde, "alpha"):
            shifted_pde.alpha *= mult
            param_desc = f"α={shifted_pde.alpha:.2e}"
        elif hasattr(shifted_pde, "diffusivity"):
            shifted_pde.diffusivity *= mult
            param_desc = f"α={shifted_pde.diffusivity:.2e}"
        elif hasattr(shifted_pde, "c"):
            shifted_pde.c *= mult
            param_desc = f"c={shifted_pde.c:.2f}"
        elif hasattr(shifted_pde, "diffusion_u"):
            shifted_pde.diffusion_u *= mult
            param_desc = f"Du={shifted_pde.diffusion_u:.2e}"

        param_data = shifted_pde.generate_dataset(n_samples=n_samples, resolution=base_resolution, seed=1003)
        param_l2, param_h1 = evaluate_on_dataset(model, param_data, device=device)
        deg = param_l2 / max(base_l2, 1e-8)
        report.cases.append(
            OODCaseResult(
                axis=OODShiftType.PARAMETER.value,
                name=f"Param ({param_desc})",
                description=f"Physical coefficient shift ×{mult:.1f}",
                in_domain_l2=base_l2,
                ood_l2=param_l2,
                degradation_ratio=deg,
                ood_h1=param_h1,
            )
        )

    # 3. Physics / Forcing OOD
    for f_mult in forcing_multipliers:
        logger.info("Running Physics/Forcing OOD test (forcing mult=%.2f)...", f_mult)
        forcing_pde = copy.deepcopy(pde)
        if hasattr(forcing_pde, "forcing_amp"):
            forcing_pde.forcing_amp *= f_mult
        forcing_data = forcing_pde.generate_dataset(n_samples=n_samples, resolution=base_resolution, seed=1004)
        f_l2, f_h1 = evaluate_on_dataset(model, forcing_data, device=device)
        deg = f_l2 / max(base_l2, 1e-8)
        report.cases.append(
            OODCaseResult(
                axis=OODShiftType.PHYSICS.value,
                name=f"Forcing ×{f_mult:.1f}",
                description=f"Source/forcing amplitude scaled ×{f_mult:.1f}",
                in_domain_l2=base_l2,
                ood_l2=f_l2,
                degradation_ratio=deg,
                ood_h1=f_h1,
            )
        )

    # 4. Geometry OOD (Warped Grid)
    logger.info("Running Geometry OOD test (non-Cartesian grid warping)...")
    warped_grid = warp_grid_coordinates(base_resolution, distortion_strength=0.06, device=device)
    geom_l2, geom_h1 = evaluate_on_dataset(model, base_data, grid=warped_grid, device=device)
    deg = geom_l2 / max(base_l2, 1e-8)
    report.cases.append(
        OODCaseResult(
            axis=OODShiftType.GEOMETRY.value,
            name="Warped Coordinates",
            description="Non-equispaced smooth grid deformation (ε=0.06)",
            in_domain_l2=base_l2,
            ood_l2=geom_l2,
            degradation_ratio=deg,
            ood_h1=geom_h1,
        )
    )

    # 5. Combined OOD (Resolution + Parameter + Geometry)
    comb_res = target_resolutions[0]
    logger.info("Running Combined OOD test (Res %d + Param shift + Warped grid)...", comb_res)
    comb_pde = copy.deepcopy(pde)
    if hasattr(comb_pde, "viscosity"):
        comb_pde.viscosity *= parameter_multipliers[0]
    elif hasattr(comb_pde, "alpha"):
        comb_pde.alpha *= parameter_multipliers[0]
    elif hasattr(comb_pde, "diffusivity"):
        comb_pde.diffusivity *= parameter_multipliers[0]

    comb_data = comb_pde.generate_dataset(n_samples=n_samples, resolution=comb_res, seed=1005)
    comb_grid = warp_grid_coordinates(comb_res, distortion_strength=0.05, device=device)
    comb_l2, comb_h1 = evaluate_on_dataset(model, comb_data, grid=comb_grid, device=device)
    deg = comb_l2 / max(base_l2, 1e-8)
    report.cases.append(
        OODCaseResult(
            axis=OODShiftType.COMBINED.value,
            name=f"Res{comb_res}+Param+Geom",
            description="Simultaneous resolution, parameter, and coordinate shift",
            in_domain_l2=base_l2,
            ood_l2=comb_l2,
            degradation_ratio=deg,
            ood_h1=comb_h1,
        )
    )

    return report

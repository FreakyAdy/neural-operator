"""Operator stress testing subsystem.

Analyzes what breaks when resolution, geometry, parameters, boundaries,
and noise change simultaneously or under deep temporal rollouts.
"""

from __future__ import annotations

from operatorlab.stress.boundary_shift import BoundaryStressResult, evaluate_boundary_stress
from operatorlab.stress.geometry_shift import GeometryStressResult, evaluate_geometry_stress
from operatorlab.stress.long_horizon import LongHorizonStressResult, evaluate_long_horizon_stress
from operatorlab.stress.noise import NoiseStressResult, evaluate_noise_stress
from operatorlab.stress.parameter_shift import ParameterStressResult, evaluate_parameter_stress
from operatorlab.stress.resolution import ResolutionStressResult, evaluate_resolution_stress

__all__ = [
    "BoundaryStressResult",
    "GeometryStressResult",
    "LongHorizonStressResult",
    "NoiseStressResult",
    "ParameterStressResult",
    "ResolutionStressResult",
    "evaluate_boundary_stress",
    "evaluate_geometry_stress",
    "evaluate_long_horizon_stress",
    "evaluate_noise_stress",
    "evaluate_parameter_stress",
    "evaluate_resolution_stress",
]

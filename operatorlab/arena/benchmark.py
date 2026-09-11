"""OperatorArena Benchmark Orchestrator.

Runs standardized competitive evaluations across multiple neural operators
evaluating accuracy, OOD generalization, robustness under perturbation,
scientific validity, and computational efficiency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from operatorlab.arena.leaderboard import LeaderboardEntry, OperatorArenaLeaderboard
from operatorlab.data.datasets import InMemoryDataset
from operatorlab.evaluation.metrics import relative_l2_error
from operatorlab.evaluation.ood import run_ood_generalization_benchmark
from operatorlab.evaluation.scientific_audit import run_scientific_audit
from operatorlab.evaluation.stress import run_stress_test
from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@dataclass
class ArenaModelMetrics:
    """Raw benchmark metrics for a single model."""

    model_name: str
    parameters: int
    latency_ms: float
    relative_l2: float
    ood_score: float
    robustness_score: float
    scientific_score: float
    validity_verdict: str
    mass_conservation: float
    arena_score: float


class ArenaBenchmark:
    """Benchmark runner for OperatorArena."""

    def __init__(
        self,
        pde: PDEProblem,
        resolution: int = 64,
        n_eval_samples: int = 50,
        device: str = "cpu",
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize Arena Benchmark.

        Args:
            pde: Target PDE problem.
            resolution: Grid resolution.
            n_eval_samples: Number of samples per test condition.
            device: Torch device.
            weights: Weight dictionary for aggregate Arena score.
                Defaults: accuracy=0.25, ood=0.25, robustness=0.25, scientific=0.15, efficiency=0.10.
        """
        self.pde = pde
        self.resolution = resolution
        self.n_eval_samples = n_eval_samples
        self.device = device
        self.weights = weights or {
            "accuracy": 0.25,
            "ood": 0.25,
            "robustness": 0.25,
            "scientific": 0.15,
            "efficiency": 0.10,
        }

    def evaluate_model(
        self,
        model: NeuralOperator,
        name: Optional[str] = None,
    ) -> ArenaModelMetrics:
        """Run all evaluation pillars on a single neural operator."""
        model_name = name or type(model).__name__
        model.to(self.device)
        model.eval()

        logger.info("Evaluating %s in OperatorArena...", model_name)

        # 1. Generate standard test dataset
        test_data = self.pde.generate_dataset(
            n_samples=self.n_eval_samples,
            resolution=self.resolution,
            seed=5555,
        )
        ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=False)
        loader = DataLoader(ds, batch_size=min(20, self.n_eval_samples))

        # Measure inference latency
        a_sample = test_data["a"][:10].to(self.device)
        n_warm = len(a_sample)
        grid_sample = None
        if hasattr(model, "make_grid"):
            grid_sample = model.make_grid((self.resolution, self.resolution), self.device).expand(n_warm, -1, -1, -1)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(a_sample, grid=grid_sample)

        t0 = time.perf_counter()
        n_iters = 10
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(a_sample, grid=grid_sample)
        t_total = time.perf_counter() - t0
        latency_ms = (t_total / max(1, n_iters * n_warm)) * 1000.0

        # In-domain L2 error
        all_preds = []
        all_u = []
        with torch.no_grad():
            for a, u, g in loader:
                a = a.to(self.device)
                g = g.to(self.device)
                if g.ndim == 3:
                    g = g.unsqueeze(0).expand(a.shape[0], -1, -1, -1)
                all_preds.append(model(a, grid=g))
                all_u.append(u.to(self.device))
        l2_err = relative_l2_error(torch.cat(all_preds, dim=0), torch.cat(all_u, dim=0)).item()

        # 2. OOD Generalization
        logger.info("Running OOD Generalization for %s...", model_name)
        ood_report = run_ood_generalization_benchmark(
            model=model,
            pde=self.pde,
            base_resolution=self.resolution,
            n_samples=min(25, self.n_eval_samples),
            device=self.device,
        )
        ood_score = ood_report.generalization_score

        # 3. Robustness Stress Test
        logger.info("Running Robustness Stress Test for %s...", model_name)
        stress_report = run_stress_test(model=model, test_loader=loader, device=self.device)
        robustness_score = stress_report.robustness_score

        # 4. Scientific Validity Audit
        logger.info("Running Scientific Validity Audit for %s...", model_name)
        audit_card = run_scientific_audit(
            model=model,
            pde=self.pde,
            test_loader=loader,
            device=self.device,
            max_rollout_steps=15,
        )
        # Scientific score: percentage of criteria passed
        n_criteria = max(len(audit_card.criteria), 1)
        n_pass = sum(1 for c in audit_card.criteria if c.status.value == "PASS")
        n_warn = sum(1 for c in audit_card.criteria if c.status.value == "WARN")
        scientific_score = round(100.0 * (n_pass + 0.5 * n_warn) / n_criteria, 1)

        mass_criterion = next((c for c in audit_card.criteria if "Mass" in c.name), None)
        mass_drift = mass_criterion.measured_value if mass_criterion else 0.0

        # Short verdict code
        if "PHYSICALLY VALID" in audit_card.overall_verdict:
            verdict = "PASS"
        elif "CONDITIONALLY VALID" in audit_card.overall_verdict:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        # 5. Composite Arena Score calculation
        # Accuracy score: 100 * exp(-10 * l2_err)
        acc_score = max(0.0, min(100.0, 100.0 * float(torch.exp(torch.tensor(-15.0 * l2_err)).item())))
        # Efficiency score: based on parameter count & latency
        params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
        eff_score = max(10.0, min(100.0, 100.0 - (params / 1e5) * 5.0 - (latency_ms / 10.0) * 5.0))

        arena_score = (
            self.weights["accuracy"] * acc_score
            + self.weights["ood"] * ood_score
            + self.weights["robustness"] * robustness_score
            + self.weights["scientific"] * scientific_score
            + self.weights["efficiency"] * eff_score
        )
        arena_score = round(arena_score, 1)

        return ArenaModelMetrics(
            model_name=model_name,
            parameters=params,
            latency_ms=latency_ms,
            relative_l2=l2_err,
            ood_score=ood_score,
            robustness_score=robustness_score,
            scientific_score=scientific_score,
            validity_verdict=verdict,
            mass_conservation=mass_drift,
            arena_score=arena_score,
        )

    def run(self, models: Dict[str, NeuralOperator]) -> OperatorArenaLeaderboard:
        """Run full competition across all submitted neural operator models."""
        pde_name = getattr(self.pde, "name", type(self.pde).__name__)
        leaderboard = OperatorArenaLeaderboard(pde_name=pde_name, resolution=self.resolution)

        for name, model in models.items():
            metrics = self.evaluate_model(model, name=name)
            leaderboard.add_entry(
                LeaderboardEntry(
                    rank=0,
                    model_name=metrics.model_name,
                    arena_score=metrics.arena_score,
                    relative_l2=metrics.relative_l2,
                    ood_score=metrics.ood_score,
                    robustness_score=metrics.robustness_score,
                    validity_verdict=metrics.validity_verdict,
                    mass_conservation=metrics.mass_conservation,
                    latency_ms=metrics.latency_ms,
                    parameters=metrics.parameters,
                )
            )

        return leaderboard

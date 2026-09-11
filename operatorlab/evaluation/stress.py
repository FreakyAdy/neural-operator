"""Operator Robustness and Stress Testing Suite.

Perturbs input fields, observation grids, and boundaries to stress-test
the stability, noise-invariance, and error tolerance of trained neural operators.

Perturbations:
- Input Gaussian noise (varying SNRs)
- Observation / sensor sparsity (missing observations, random dropouts)
- Coordinate jitter / irregular point displacement
- High-frequency spectral truncation (band-limited inputs)
- Boundary perturbations (localized boundary noise)
- Shot / impulsive noise (sensor faults)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from operatorlab.evaluation.metrics import h1_error, relative_l2_error
from operatorlab.models.base import NeuralOperator

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Perturbation Transforms
# -----------------------------------------------------------------------------

def perturb_gaussian_noise(a: Tensor, sigma: float = 0.05, relative: bool = True) -> Tensor:
    """Add zero-mean Gaussian noise to input tensor.

    Args:
        a: Input tensor, shape (batch, ..., channels).
        sigma: Noise standard deviation. If relative=True, scaled by standard deviation of a.
        relative: Whether sigma is relative to a's standard deviation.
    """
    if relative:
        scale = a.std() * sigma
    else:
        scale = sigma
    noise = torch.randn_like(a) * scale
    return a + noise


def perturb_sensor_sparsity(
    a: Tensor,
    keep_ratio: float = 0.5,
    mask_mode: str = "zero",
) -> Tensor:
    """Simulate missing sensor observations by dropping random spatial points.

    Args:
        a: Input tensor, shape (batch, H, W, channels).
        keep_ratio: Fraction of spatial locations retained (e.g. 0.25 = 75% missing).
        mask_mode: 'zero' to fill missing with 0, or 'mean' to fill with mean.
    """
    batch_size, h, w, c = a.shape
    # Spatial mask shared across channels
    mask = (torch.rand(batch_size, h, w, 1, device=a.device) < keep_ratio).float()
    if mask_mode == "mean":
        mean_val = a.mean(dim=(1, 2), keepdim=True)
        return a * mask + mean_val * (1.0 - mask)
    return a * mask


def perturb_spectral_truncation(a: Tensor, keep_ratio: float = 0.5) -> Tensor:
    """Low-pass filter input field by truncating high-frequency 2D Fourier modes.

    Args:
        a: Input tensor, shape (batch, H, W, channels).
        keep_ratio: Fraction of lower modes to retain (e.g. 0.5 retains bottom 50% frequencies).
    """
    batch, h, w, c = a.shape
    # FFT over spatial dimensions
    a_perm = a.permute(0, 3, 1, 2)  # (batch, c, H, W)
    a_hat = torch.fft.rfft2(a_perm)

    freq_h, freq_w = a_hat.shape[-2:]
    cutoff_h = max(1, int(freq_h * keep_ratio))
    cutoff_w = max(1, int(freq_w * keep_ratio))

    # Zero out frequencies beyond cutoff
    filtered_hat = torch.zeros_like(a_hat)
    filtered_hat[..., :cutoff_h, :cutoff_w] = a_hat[..., :cutoff_h, :cutoff_w]

    filtered = torch.fft.irfft2(filtered_hat, s=(h, w))
    return filtered.permute(0, 2, 3, 1)


def perturb_coordinate_jitter(
    grid: Tensor,
    jitter_std: float = 0.02,
) -> Tensor:
    """Perturb grid coordinates with random jitter while preserving domain bounds [0, 1].

    Args:
        grid: Coordinate tensor, shape (batch, H, W, 2).
        jitter_std: Standard deviation of coordinate jitter.
    """
    noise = torch.randn_like(grid) * jitter_std
    return torch.clamp(grid + noise, 0.0, 1.0)


def perturb_boundary_noise(
    a: Tensor,
    boundary_width: int = 3,
    sigma: float = 0.15,
) -> Tensor:
    """Inject noise specifically along the boundary perimeter of the input field.

    Args:
        a: Input field, shape (batch, H, W, channels).
        boundary_width: Number of grid cells bordering the perimeter.
        sigma: Standard deviation of boundary noise.
    """
    batch, h, w, c = a.shape
    mask = torch.zeros(batch, h, w, 1, device=a.device)
    mask[:, :boundary_width, :, :] = 1.0
    mask[:, -boundary_width:, :, :] = 1.0
    mask[:, :, :boundary_width, :] = 1.0
    mask[:, :, -boundary_width:, :] = 1.0

    noise = torch.randn_like(a) * (a.std() * sigma)
    return a + noise * mask


def perturb_shot_noise(a: Tensor, p: float = 0.02, scale: float = 2.0) -> Tensor:
    """Inject impulsive shot noise / sensor fault spikes.

    Args:
        a: Input field.
        p: Probability of a point being corrupted with a spike.
        scale: Amplitude multiplier relative to data std.
    """
    std = a.std()
    base_scale = std * scale if std > 1e-6 else scale
    mask = (torch.rand_like(a) < p).float()
    spikes = (torch.randn_like(a).sign()) * base_scale
    return a + mask * spikes


# -----------------------------------------------------------------------------
# Evaluation Engine & Reports
# -----------------------------------------------------------------------------

@dataclass
class StressConditionResult:
    """Evaluation result under a specific stress condition."""

    name: str
    category: str
    severity: str
    clean_l2: float
    perturbed_l2: float
    degradation_ratio: float
    perturbed_h1: float = 0.0
    is_stable: bool = True  # whether output was non-NaN and bounded


@dataclass
class OperatorRobustnessReport:
    """Report summarizing robustness testing for a single neural operator."""

    model_name: str
    clean_l2: float
    results: list[StressConditionResult] = field(default_factory=list)

    @property
    def robustness_score(self) -> float:
        """Overall robustness score between 0 and 100.

        100 = completely impervious to noise/sparsity.
        Penalizes degradation ratios > 1.0 and unstable predictions.
        """
        if not self.results:
            return 100.0
        scores = []
        for r in self.results:
            if not r.is_stable:
                scores.append(0.0)
            else:
                deg = max(1.0, r.degradation_ratio)
                s = 100.0 / (1.0 + 0.6 * (deg - 1.0))
                scores.append(s)
        return round(sum(scores) / len(scores), 2)

    def summary_table(self) -> str:
        """Format an executive terminal table."""
        lines = [
            f"=== OPERATOR ROBUSTNESS REPORT: {self.model_name} ===",
            f"Baseline Clean L2: {self.clean_l2:.6f} | Robustness Score: {self.robustness_score}/100",
            "-" * 84,
            f"{'Test Condition':<22} | {'Category':<14} | {'Severity':<10} | {'Perturbed L2':<14} | {'Degradation':<11} | {'Status':<6}",
            "-" * 84,
        ]
        for r in self.results:
            status = "PASS" if r.is_stable and r.degradation_ratio < 3.0 else ("WARN" if r.is_stable else "FAIL")
            lines.append(
                f"{r.name:<22} | {r.category:<14} | {r.severity:<10} | {r.perturbed_l2:<14.6f} | {r.degradation_ratio:.2f}x       | {status:<6}"
            )
        lines.append("-" * 84)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "clean_l2": self.clean_l2,
            "robustness_score": self.robustness_score,
            "results": [asdict(r) for r in self.results],
        }


@dataclass
class MultiModelRobustnessReport:
    """Side-by-side comparison of multiple neural operators under stress."""

    models: list[str]
    clean_errors: dict[str, float]
    # condition_name -> {model_name -> perturbed_l2}
    condition_errors: dict[str, dict[str, float]]
    robustness_scores: dict[str, float]

    def summary_table(self) -> str:
        """Render the cross-model robustness matrix."""
        col_w = 12
        header = f"{'Condition':<22} | " + " | ".join(f"{m:<{col_w}}" for m in self.models)
        sep = "-" * len(header)
        lines = [
            "=================== OPERATOR ROBUSTNESS COMPARISON ===================",
            header,
            sep,
        ]

        # Clean row
        clean_row = f"{'Clean (Baseline)':<22} | " + " | ".join(
            f"{self.clean_errors.get(m, 0.0):<{col_w}.4f}" for m in self.models
        )
        lines.append(clean_row)
        lines.append(sep)

        for cond, model_vals in self.condition_errors.items():
            row = f"{cond:<22} | " + " | ".join(
                f"{model_vals.get(m, 0.0):<{col_w}.4f}" for m in self.models
            )
            lines.append(row)

        lines.append(sep)
        score_row = f"{'Robustness Score':<22} | " + " | ".join(
            f"{self.robustness_scores.get(m, 0.0):<{col_w}.1f}" for m in self.models
        )
        lines.append(score_row)
        lines.append("=" * len(header))
        return "\n".join(lines)


@torch.no_grad()
def run_stress_test(
    model: NeuralOperator,
    test_loader: DataLoader,
    device: str = "cpu",
) -> OperatorRobustnessReport:
    """Run full operational stress testing suite on a single neural operator.

    Args:
        model: Trained neural operator.
        test_loader: DataLoader yielding (a, u, grid) batches.
        device: Device to run evaluation on.

    Returns:
        OperatorRobustnessReport containing all stress condition outcomes.
    """
    model.eval()
    model_name = type(model).__name__

    # 1. Clean evaluation
    clean_preds = []
    all_u = []
    all_a = []
    all_grid = []

    for a, u, grid in test_loader:
        a = a.to(device)
        u = u.to(device)
        grid = grid.to(device)
        if grid.ndim == 3:
            grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)

        pred = model(a, grid=grid)
        clean_preds.append(pred)
        all_u.append(u)
        all_a.append(a)
        all_grid.append(grid)

    clean_u_cat = torch.cat(all_u, dim=0)
    clean_pred_cat = torch.cat(clean_preds, dim=0)
    clean_l2 = relative_l2_error(clean_pred_cat, clean_u_cat).item()

    report = OperatorRobustnessReport(model_name=model_name, clean_l2=clean_l2)

    # Define stress test suite
    test_suite = [
        # Noise
        ("Gaussian Noise (Low)", "Noise", "σ=0.02", lambda a, g: (perturb_gaussian_noise(a, sigma=0.02), g)),
        ("Gaussian Noise (Med)", "Noise", "σ=0.05", lambda a, g: (perturb_gaussian_noise(a, sigma=0.05), g)),
        ("Gaussian Noise (High)", "Noise", "σ=0.10", lambda a, g: (perturb_gaussian_noise(a, sigma=0.10), g)),
        ("Impulsive Spikes", "Fault", "p=0.02", lambda a, g: (perturb_shot_noise(a, p=0.02), g)),
        # Sparsity
        ("Sensor Sparsity 75%", "Sparsity", "keep 75%", lambda a, g: (perturb_sensor_sparsity(a, keep_ratio=0.75), g)),
        ("Sensor Sparsity 50%", "Sparsity", "keep 50%", lambda a, g: (perturb_sensor_sparsity(a, keep_ratio=0.50), g)),
        ("Sensor Sparsity 25%", "Sparsity", "keep 25%", lambda a, g: (perturb_sensor_sparsity(a, keep_ratio=0.25), g)),
        # Spectral
        ("Spectral Cut 50%", "Frequency", "lowpass 50%", lambda a, g: (perturb_spectral_truncation(a, keep_ratio=0.50), g)),
        ("Spectral Cut 25%", "Frequency", "lowpass 25%", lambda a, g: (perturb_spectral_truncation(a, keep_ratio=0.25), g)),
        # Geometry / Jitter
        ("Coord Jitter (Low)", "Mesh", "std=0.01", lambda a, g: (a, perturb_coordinate_jitter(g, jitter_std=0.01))),
        ("Coord Jitter (Med)", "Mesh", "std=0.03", lambda a, g: (a, perturb_coordinate_jitter(g, jitter_std=0.03))),
        # Boundary
        ("Boundary Noise", "Boundary", "σ=0.15", lambda a, g: (perturb_boundary_noise(a, boundary_width=2, sigma=0.15), g)),
    ]

    for name, category, severity, perturb_fn in test_suite:
        perturbed_preds = []
        is_stable = True

        for a, grid in zip(all_a, all_grid):
            p_a, p_grid = perturb_fn(a, grid)
            pred = model(p_a, grid=p_grid)

            if torch.isnan(pred).any() or torch.isinf(pred).any() or pred.abs().max() > 1e4:
                is_stable = False
                break
            perturbed_preds.append(pred)

        if not is_stable or len(perturbed_preds) == 0:
            res = StressConditionResult(
                name=name,
                category=category,
                severity=severity,
                clean_l2=clean_l2,
                perturbed_l2=float("nan"),
                degradation_ratio=float("inf"),
                is_stable=False,
            )
        else:
            cat_preds = torch.cat(perturbed_preds, dim=0)
            p_l2 = relative_l2_error(cat_preds, clean_u_cat).item()
            h = clean_u_cat.shape[1]
            p_h1 = h1_error(cat_preds, clean_u_cat, dx=1.0 / h).item()
            deg = p_l2 / max(clean_l2, 1e-8)

            res = StressConditionResult(
                name=name,
                category=category,
                severity=severity,
                clean_l2=clean_l2,
                perturbed_l2=p_l2,
                degradation_ratio=deg,
                perturbed_h1=p_h1,
                is_stable=True,
            )

        report.results.append(res)

    return report

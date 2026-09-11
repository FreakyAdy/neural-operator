"""Lightweight JSON-based experiment tracking.

No external dependencies (no MLflow, no W&B). Stores configs, per-epoch
metrics, and final evaluation results as JSON files in a structured directory.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ExperimentRegistry:
    """Lightweight experiment tracker.

    Stores:
        - Full resolved config
        - Metrics per epoch (JSONL format)
        - Final evaluation results
        - Git hash and timestamp

    Directory structure:
        experiments/{run_name}/
        ├── config.yaml
        ├── metrics.jsonl
        ├── eval.json
        ├── checkpoints/
        │   ├── best.pt
        │   └── last.pt
        └── plots/
            ├── training_curve.png
            ├── resolution_sweep.png
            └── error_spectrum.png

    Args:
        run_name: Name of the experiment run.
        output_dir: Base directory for experiments.
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path = "experiments/",
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{run_name}_{timestamp}"
        self.run_dir = Path(output_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._eval_path = self.run_dir / "eval.json"
        self._config_path = self.run_dir / "config.json"

        logger.info("Experiment registry initialized: %s", self.run_dir)

    def log_config(self, config: dict) -> None:
        """Save the full experiment configuration.

        Args:
            config: Configuration dictionary (from ExperimentConfig.to_dict()).
        """
        config_with_meta = {
            **config,
            "_git_hash": _get_git_hash(),
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "_run_name": self.run_name,
        }
        with open(self._config_path, "w") as f:
            json.dump(config_with_meta, f, indent=2, default=str)
        logger.debug("Saved config to %s", self._config_path)

    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Append per-epoch metrics.

        Args:
            epoch: Current epoch number.
            metrics: Dict of metric names to values.
        """
        entry = {"epoch": epoch, **metrics}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_eval(self, eval_result: dict) -> None:
        """Save final evaluation results.

        Args:
            eval_result: Evaluation results dictionary.
        """
        with open(self._eval_path, "w") as f:
            json.dump(eval_result, f, indent=2, default=str)
        logger.info("Saved evaluation results to %s", self._eval_path)

    def summary(self) -> dict:
        """Load and return a summary of the experiment.

        Returns:
            Dict with config, final metrics, and eval results.
        """
        summary = {"run_name": self.run_name, "run_dir": str(self.run_dir)}

        if self._config_path.exists():
            with open(self._config_path) as f:
                summary["config"] = json.load(f)

        if self._metrics_path.exists():
            metrics = []
            with open(self._metrics_path) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
            summary["n_epochs"] = len(metrics)
            if metrics:
                summary["final_metrics"] = metrics[-1]
                summary["best_val_loss"] = min(
                    (m.get("val_loss", float("inf")) for m in metrics),
                )

        if self._eval_path.exists():
            with open(self._eval_path) as f:
                summary["eval"] = json.load(f)

        return summary

    @property
    def checkpoint_dir(self) -> Path:
        """Path to the checkpoints directory."""
        return self.run_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        """Path to the plots directory."""
        return self.run_dir / "plots"

"""Unit tests for experiment registry and tracking."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from operatorlab.registry.experiment import ExperimentRegistry


class TestExperimentRegistry:
    def test_registry_lifecycle(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(run_name="test_run", output_dir=tmp_path)
        assert reg.run_dir.exists()
        assert reg.checkpoint_dir.exists()
        assert reg.plots_dir.exists()

        # Log config
        config = {"model": {"type": "fno"}, "training": {"lr": 0.001}}
        reg.log_config(config)
        assert (reg.run_dir / "config.json").exists()

        # Log metrics
        reg.log_metrics(1, {"train_loss": 0.5, "val_loss": 0.4})
        reg.log_metrics(2, {"train_loss": 0.3, "val_loss": 0.2})
        assert (reg.run_dir / "metrics.jsonl").exists()

        # Log evaluation
        reg.log_eval({"l2_error": 0.05, "h1_error": 0.08})
        assert (reg.run_dir / "eval.json").exists()

        # Check summary
        summ = reg.summary()
        assert summ["n_epochs"] == 2
        assert summ["best_val_loss"] == pytest.approx(0.2)
        assert summ["eval"]["l2_error"] == pytest.approx(0.05)

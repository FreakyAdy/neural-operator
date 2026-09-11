"""Integration tests for benchmark scripts."""

from __future__ import annotations

import subprocess
import sys


def test_benchmark_fno_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_fno.py",
            "--device", "cpu",
            "--resolutions", "16", "32",
            "--batch-size", "2",
            "--n-warmup", "1",
            "--n-runs", "1",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "FNO2d parameters" in result.stdout or "Resolution" in result.stdout


def test_physics_violation_audit_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/physics_violation_audit.py",
            "--device", "cpu",
            "--epochs", "1",
            "--resolution", "16",
            "--n-train", "4",
            "--n-test", "4",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "Physics Violation Audit" in result.stdout


def test_resolution_generalization_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/resolution_generalization.py",
            "--device", "cpu",
            "--epochs", "1",
            "--resolution", "16",
            "--n-train", "4",
            "--n-test", "4",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "Resolution" in result.stdout


def test_architecture_comparison_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/architecture_comparison.py",
            "--device", "cpu",
            "--epochs", "1",
            "--resolution", "16",
            "--n-train", "4",
            "--n-test", "4",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "FNO" in result.stdout
    assert "TFNO" in result.stdout
    assert "DeepONet" in result.stdout

"""Integration tests for the CLI commands."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    """CLI --help returns exit code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "operatorlab" in result.stdout.lower() or "Usage" in result.stdout


def test_list_models() -> None:
    """list-models command returns exit code 0 and lists architectures."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "list-models"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "fno" in result.stdout.lower()
    assert "tfno" in result.stdout.lower()
    assert "deeponet" in result.stdout.lower()
    assert "gno" in result.stdout.lower()
    assert "hybrid" in result.stdout.lower()


def test_list_pdes() -> None:
    """list-pdes command returns exit code 0 and lists PDEs."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "list-pdes"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "heat" in result.stdout.lower()
    assert "navier_stokes" in result.stdout.lower()
    assert "wave" in result.stdout.lower()


def test_generate_help() -> None:
    """Generate subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "generate", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_train_help() -> None:
    """Train subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "train", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_evaluate_help() -> None:
    """Evaluate subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "evaluate", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_compare_help() -> None:
    """Compare subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "compare", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_visualize_help() -> None:
    """Visualize subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "visualize", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0


def test_ood_help() -> None:
    """OOD subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "ood", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "out-of-distribution" in result.stdout.lower() or "ood" in result.stdout.lower()


def test_stress_help() -> None:
    """Stress subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "stress", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "robustness" in result.stdout.lower() or "stress" in result.stdout.lower()


def test_audit_help() -> None:
    """Audit subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "audit", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "scientific validity" in result.stdout.lower() or "audit" in result.stdout.lower()


def test_arena_help() -> None:
    """Arena subcommand --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "operatorlab.cli", "arena", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "arena" in result.stdout.lower() or "leaderboard" in result.stdout.lower()

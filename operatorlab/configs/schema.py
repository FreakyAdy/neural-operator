"""Configuration schema using Python dataclasses.

All experiment configuration is defined here. YAML configs are loaded
and validated into these dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Neural operator model hyperparameters."""

    type: str = "fno"
    modes: int = 12
    width: int = 64
    n_layers: int = 4
    input_dim: int = 1
    output_dim: int = 1

    # DeepONet-specific
    n_sensors: int = 64 * 64
    branch_depth: int = 4
    trunk_depth: int = 4
    basis_functions: int = 128

    # GNO-specific
    n_neighbors: int = 16
    edge_features: int = 4

    # Hybrid-specific
    attention_heads: int = 4
    attention_layers: int = 2

    # TFNO-specific
    rank: int = 16


@dataclass
class LossConfig:
    """Loss function weights."""

    l2_weight: float = 1.0
    h1_weight: float = 0.0
    pde_weight: float = 0.0


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    epochs: int = 500
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    amp: bool = True
    grad_clip: float = 1.0
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class PDEConfig:
    """PDE problem definition."""

    type: str = "heat"
    viscosity: float = 1e-3
    T: float = 1.0
    resolution: int = 64
    alpha: float = 0.01  # thermal diffusivity (heat equation)
    c: float = 1.0  # wave speed
    dt: float = 1e-3


@dataclass
class DataConfig:
    """Dataset generation and loading parameters."""

    n_train: int = 1000
    n_val: int = 200
    n_test: int = 200
    batch_size: int = 20
    cache_dir: str = "data/"
    seed: int = 42
    num_workers: int = 0


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    resolution_sweep: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    eval_every: int = 50
    n_test_samples: int = 200


@dataclass
class ExperimentMeta:
    """Experiment metadata."""

    name: str = "experiment"
    seed: int = 42
    output_dir: str = "experiments/"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Bundles model, PDE, data, training, evaluation, and experiment metadata.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict."""
        return asdict(self)


def _merge_dataclass(dc_class: type, raw: dict) -> object:
    """Recursively build a dataclass from a raw dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(dc_class)}
    filtered = {}
    for key, value in raw.items():
        if key not in field_names:
            logger.warning("Ignoring unknown config key: %s", key)
            continue
        f = next(f for f in dataclasses.fields(dc_class) if f.name == key)
        if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else None):
            value = _merge_dataclass(f.type, value)
        elif hasattr(f.type, "__origin__"):
            # Handle generic types like list[int] — just pass through
            pass
        filtered[key] = value
    return dc_class(**filtered)


def from_yaml(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated ExperimentConfig.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return ExperimentConfig()

    config = ExperimentConfig()

    section_map = {
        "model": (ModelConfig, "model"),
        "pde": (PDEConfig, "pde"),
        "data": (DataConfig, "data"),
        "training": (TrainingConfig, "training"),
        "evaluation": (EvalConfig, "evaluation"),
        "experiment": (ExperimentMeta, "experiment"),
    }

    for section_key, (dc_class, attr_name) in section_map.items():
        if section_key in raw and isinstance(raw[section_key], dict):
            section_data = raw[section_key]
            # Handle nested dataclasses (e.g., training.loss)
            if dc_class is TrainingConfig and "loss" in section_data:
                section_data = dict(section_data)
                section_data["loss"] = _merge_dataclass(LossConfig, section_data["loss"])
            setattr(config, attr_name, _merge_dataclass(dc_class, section_data))

    return config

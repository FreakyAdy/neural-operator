"""Learning rate scheduler factory.

Builds a scheduler from a config string. Supports: cosine, step, plateau.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    epochs: int,
    **kwargs,
) -> LRScheduler | None:
    """Build a learning rate scheduler from a config name.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler name: 'cosine', 'step', 'plateau', or 'none'.
        epochs: Total number of training epochs.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        Configured LR scheduler, or None if name is 'none'.
    """
    name = name.lower()

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, **kwargs)
    elif name == "step":
        step_size = kwargs.pop("step_size", max(epochs // 3, 1))
        gamma = kwargs.pop("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, patience=10, factor=0.5, **kwargs)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}. Choose from: cosine, step, plateau, none")

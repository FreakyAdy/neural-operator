"""Integration test: full training smoke test.

Train for 3 steps on tiny data, verify loss decreases.
Should complete in < 10 seconds on CPU.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from operatorlab.configs.schema import (
    DataConfig, ExperimentConfig, ExperimentMeta,
    EvalConfig, ModelConfig, PDEConfig, TrainingConfig,
)
from operatorlab.data.datasets import InMemoryDataset
from operatorlab.models.fno import FNO2d
from operatorlab.training.losses import OperatorLoss
from operatorlab.training.trainer import Trainer


def test_full_training_smoke() -> None:
    """Train FNO for 3 epochs on tiny heat data, verify loss decreases."""
    from operatorlab.physics.heat import HeatEquation2D

    pde = HeatEquation2D(alpha=0.01, T=1.0)
    data = pde.generate_dataset(n_samples=20, resolution=16, seed=42)

    train_ds = InMemoryDataset(data["a"][:15], data["u"][:15], normalize=True)
    val_ds = InMemoryDataset(data["a"][15:], data["u"][15:], normalize=True)

    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=5)

    model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = OperatorLoss(l2_weight=1.0)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        use_amp=False,
        grad_clip=1.0,
    )

    result = trainer.train(train_loader, val_loader, n_epochs=3)

    assert result.final_train_loss < result.initial_train_loss, \
        f"Loss did not decrease: {result.initial_train_loss:.4f} → {result.final_train_loss:.4f}"
    assert len(result.train_losses) == 3
    assert len(result.val_losses) == 3


def test_training_with_h1_loss() -> None:
    """Train with H1 loss enabled (small weight to avoid destabilizing tiny data)."""
    from operatorlab.physics.heat import HeatEquation2D

    pde = HeatEquation2D(alpha=0.01, T=1.0)
    data = pde.generate_dataset(n_samples=20, resolution=16, seed=42)

    train_ds = InMemoryDataset(data["a"][:15], data["u"][:15])
    val_ds = InMemoryDataset(data["a"][15:], data["u"][15:])

    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=5)

    model = FNO2d(modes1=4, modes2=4, width=8, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = OperatorLoss(l2_weight=1.0, h1_weight=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        use_amp=False,
        grad_clip=1.0,
    )

    result = trainer.train(train_loader, val_loader, n_epochs=5)
    # With H1 loss, convergence is slower; just check it doesn't diverge
    assert not any(
        loss != loss for loss in result.train_losses  # NaN check
    ), "Training produced NaN losses"

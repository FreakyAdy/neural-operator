"""Resolution generalization benchmark.

Train FNO at base resolution, evaluate zero-shot across resolutions.
Reports the resolution scaling table.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolution Generalization Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--resolution", type=int, default=64, help="Base training resolution")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.evaluation.resolution_sweep import resolution_sweep
    from operatorlab.models.fno import FNO2d
    from operatorlab.physics.navier_stokes import NavierStokes2D
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.trainer import Trainer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Setup
    pde = NavierStokes2D(viscosity=1e-3, T=1.0)
    train_res = args.resolution

    logger.info("Generating training data at %d×%d...", train_res, train_res)
    train_data = pde.generate_dataset(n_samples=args.n_train, resolution=train_res, seed=42)
    val_data = pde.generate_dataset(n_samples=args.n_test, resolution=train_res, seed=43)

    train_ds = InMemoryDataset(train_data["a"], train_data["u"], normalize=True)
    val_ds = InMemoryDataset(val_data["a"], val_data["u"], normalize=True)

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=20)

    # Build model
    modes = min(12, train_res // 2)
    model = FNO2d(modes1=modes, modes2=modes, width=64, n_layers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = OperatorLoss(l2_weight=1.0, h1_weight=0.1)

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        device=device, use_amp=(device != "cpu"), grad_clip=1.0,
    )

    # Train
    logger.info("Training FNO for %d epochs...", args.epochs)
    result = trainer.train(train_loader, val_loader, n_epochs=args.epochs)
    logger.info("Training done. Best val loss: %.6f", result.best_val_loss)

    # Resolution sweep
    target_res = [train_res, train_res * 2] if train_res >= 64 else [16, 32, 64]
    logger.info("Running resolution sweep on resolutions: %s...", target_res)
    sweep = resolution_sweep(
        model=model, pde=pde,
        base_resolution=train_res,
        target_resolutions=target_res,
        n_test_samples=args.n_test,
        device=device,
    )

    print("\n" + sweep.summary_table())


if __name__ == "__main__":
    main()

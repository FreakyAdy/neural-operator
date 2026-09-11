"""Architecture comparison benchmark.

Trains FNO, TFNO, and DeepONet on the same PDE data
and compares their parameter count, validation loss, and relative L2 error.
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture Comparison Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--resolution", type=int, default=64, help="Spatial resolution")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.models.fno import FNO2d
    from operatorlab.models.tfno import TFNO2d
    from operatorlab.models.deeponet import DeepONet
    from operatorlab.physics.navier_stokes import NavierStokes2D
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.trainer import Trainer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pde = NavierStokes2D(viscosity=1e-3, T=1.0)
    res = args.resolution

    # Generate data
    train_data = pde.generate_dataset(n_samples=args.n_train, resolution=res, seed=42)
    val_data = pde.generate_dataset(n_samples=args.n_test, resolution=res, seed=43)

    train_ds = InMemoryDataset(train_data["a"], train_data["u"], normalize=True)
    val_ds = InMemoryDataset(val_data["a"], val_data["u"], normalize=True)

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=20)

    modes = min(12, res // 2)
    models = {
        "FNO": FNO2d(modes1=modes, modes2=modes, width=64, n_layers=4),
        "TFNO": TFNO2d(modes1=modes, modes2=modes, width=64, n_layers=4, rank=16),
        "DeepONet": DeepONet(
            branch_input_dim=res * res, hidden_dim=128, n_basis=128,
            branch_depth=4, trunk_depth=4,
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info("Training %s (%s params)...", name, f"{model.count_parameters():,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = OperatorLoss(l2_weight=1.0)

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn,
            device=device, use_amp=(device != "cpu"), grad_clip=1.0,
        )

        result = trainer.train(train_loader, val_loader, n_epochs=args.epochs)
        eval_result = trainer.evaluate(val_loader)

        results[name] = {
            "params": model.count_parameters(),
            "best_val_loss": result.best_val_loss,
            "final_l2": eval_result.l2_error,
        }

    # Print comparison
    print(f"\n{'Model':<12} | {'Params':<10} | {'Best Val Loss':<14} | {'L2 Error':<10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<12} | {r['params']:<10,} | {r['best_val_loss']:<14.6f} | {r['final_l2']:<10.6f}")


if __name__ == "__main__":
    main()

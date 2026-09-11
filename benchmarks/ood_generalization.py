"""Out-of-Distribution (OOD) Generalization Benchmark.

Evaluates how far neural operators generalize across:
1. Resolution OOD
2. Parameter OOD (viscosity / coefficients)
3. Physics / Forcing OOD
4. Geometry OOD (warped coordinates)
5. Combined OOD
"""

from __future__ import annotations

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD Operator Generalization Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--resolution", type=int, default=64, help="Base training resolution")
    parser.add_argument("--n-samples", type=int, default=30, help="Test samples per OOD condition")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs if training from scratch")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.evaluation.ood import run_ood_generalization_benchmark
    from operatorlab.models.fno import FNO2d
    from operatorlab.physics.navier_stokes import NavierStokes2D
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.trainer import Trainer
    from torch.utils.data import DataLoader

    res = args.resolution
    pde = NavierStokes2D(viscosity=1e-3, T=1.0)

    logger.info("Generating training data for quick baseline model (%d samples)...", 100)
    train_data = pde.generate_dataset(n_samples=100, resolution=res, seed=42)
    val_data = pde.generate_dataset(n_samples=30, resolution=res, seed=43)

    train_ds = InMemoryDataset(train_data["a"], train_data["u"], normalize=True)
    val_ds = InMemoryDataset(val_data["a"], val_data["u"], normalize=True)
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=20)

    modes = min(12, res // 2)
    model = FNO2d(modes1=modes, modes2=modes, width=64, n_layers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = OperatorLoss(l2_weight=1.0)
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    logger.info("Training FNO for %d epochs...", args.epochs)
    trainer.train(train_loader, val_loader, n_epochs=args.epochs)

    logger.info("Running OOD Generalization Suite...")
    report = run_ood_generalization_benchmark(
        model=model,
        pde=pde,
        base_resolution=res,
        n_samples=args.n_samples,
        device=device,
    )

    print("\n" + report.summary_table())


if __name__ == "__main__":
    main()

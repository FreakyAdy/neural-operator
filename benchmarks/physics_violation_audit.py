"""Physics violation audit benchmark.

Evaluates how well trained models satisfy PDE constraints
and conservation laws.
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics Violation Audit Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--resolution", type=int, default=64, help="Spatial resolution")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.evaluation.physics_violation import analyze_physics_violations
    from operatorlab.models.fno import FNO2d
    from operatorlab.physics.navier_stokes import NavierStokes2D
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.trainer import Trainer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pde = NavierStokes2D(viscosity=1e-3, T=1.0)
    res = args.resolution

    # Generate data and train a model
    train_data = pde.generate_dataset(n_samples=args.n_train, resolution=res, seed=42)
    test_data = pde.generate_dataset(n_samples=args.n_test, resolution=res, seed=99)

    train_ds = InMemoryDataset(train_data["a"], train_data["u"], normalize=True)
    test_ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=True)

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=20)

    modes = min(12, res // 2)
    model = FNO2d(modes1=modes, modes2=modes, width=64, n_layers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = OperatorLoss(l2_weight=1.0)

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        device=device, use_amp=(device != "cpu"),
    )

    logger.info("Training model for audit (%d epochs)...", args.epochs)
    trainer.train(train_loader, test_loader, n_epochs=args.epochs)

    logger.info("Running physics violation audit...")
    report = analyze_physics_violations(model, pde, test_loader, device=device)

    print(f"\n--- Physics Violation Audit ---")
    print(f"Mean residual norm:  {report.mean_residual_norm:.6f}")
    print(f"Max residual norm:   {report.max_residual_norm:.6f}")
    print(f"Energy spectrum err: {report.energy_spectrum_error:.6f}")
    print(f"Divergence norm:     {report.divergence_norm:.6f}")
    print(f"Conservation violations:")
    for k, v in report.conservation_violations.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

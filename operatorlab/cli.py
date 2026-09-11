"""CLI entry point for OperatorLab.

All heavy imports (torch, etc.) are lazy-loaded inside command functions
to keep `operatorlab --help` fast (< 200ms).

Commands:
    train       Train a neural operator from a YAML config
    evaluate    Evaluate a trained checkpoint
    compare     Compare multiple checkpoints
    visualize   Generate visualizations
    generate    Generate PDE training data
    ood         Run Out-of-Distribution (OOD) operator generalization benchmark
    stress      Run operator robustness and perturbation stress tests
    audit       Audit scientific validity and physical invariants
    arena       Run OperatorArena multi-model competitive benchmark
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="operatorlab",
    help="Scientific Generalization & Robustness Laboratory for Neural Operators.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_model(model_config):
    """Build a neural operator model from config."""
    import torch
    from operatorlab.configs.schema import ModelConfig

    model_type = model_config.type.lower()

    if model_type == "fno":
        from operatorlab.models.fno import FNO2d
        return FNO2d(
            modes1=model_config.modes,
            modes2=model_config.modes,
            width=model_config.width,
            n_layers=model_config.n_layers,
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
        )
    elif model_type == "tfno":
        from operatorlab.models.tfno import TFNO2d
        return TFNO2d(
            modes1=model_config.modes,
            modes2=model_config.modes,
            width=model_config.width,
            n_layers=model_config.n_layers,
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
            rank=model_config.rank,
        )
    elif model_type == "deeponet":
        from operatorlab.models.deeponet import DeepONet
        return DeepONet(
            branch_input_dim=model_config.n_sensors,
            trunk_input_dim=2,
            hidden_dim=model_config.width,
            output_dim=model_config.output_dim,
            branch_depth=model_config.branch_depth,
            trunk_depth=model_config.trunk_depth,
            n_basis=model_config.basis_functions,
        )
    elif model_type == "gno":
        from operatorlab.models.gno import GNO
        return GNO(
            input_dim=model_config.input_dim + 2,
            hidden_dim=model_config.width,
            output_dim=model_config.output_dim,
            n_layers=model_config.n_layers,
            n_neighbors=model_config.n_neighbors,
        )
    elif model_type == "hybrid":
        from operatorlab.models.hybrid import HybridOperator
        return HybridOperator(
            modes1=model_config.modes,
            modes2=model_config.modes,
            width=model_config.width,
            n_fno_layers=model_config.n_layers,
            n_attention_layers=model_config.attention_layers,
            attention_heads=model_config.attention_heads,
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Choose from: fno, tfno, deeponet, gno, hybrid"
        )


def _build_pde(pde_config):
    """Build a PDE problem from config."""
    pde_type = pde_config.type.lower()

    if pde_type == "heat":
        from operatorlab.physics.heat import HeatEquation2D
        return HeatEquation2D(alpha=pde_config.alpha, T=pde_config.T)
    elif pde_type == "navier_stokes":
        from operatorlab.physics.navier_stokes import NavierStokes2D
        return NavierStokes2D(viscosity=pde_config.viscosity, T=pde_config.T)
    elif pde_type == "wave":
        from operatorlab.physics.wave import WaveEquation2D
        return WaveEquation2D(c=pde_config.c, T=pde_config.T)
    elif pde_type == "reaction_diffusion":
        from operatorlab.physics.reaction_diffusion import ReactionDiffusion2D
        return ReactionDiffusion2D(T=pde_config.T)
    elif pde_type == "shallow_water":
        from operatorlab.physics.shallow_water import ShallowWater2D
        return ShallowWater2D(T=pde_config.T)
    elif pde_type == "elasticity":
        from operatorlab.physics.elasticity import Elasticity2D
        return Elasticity2D()
    else:
        raise ValueError(
            f"Unknown PDE type: {pde_type}. "
            "Choose from: heat, navier_stokes, wave, reaction_diffusion, "
            "shallow_water, elasticity"
        )


@app.command(name="list-models")
def list_models() -> None:
    """List available neural operator architectures."""
    table = Table(title="Available Neural Operator Architectures")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Resolution Invariant", style="green")
    table.add_row("fno", "2D Fourier Neural Operator with real FFT spectral convs", "Yes")
    table.add_row("tfno", "Tensorized FNO using Tucker decomposed complex weight tensors", "Yes")
    table.add_row("deeponet", "Deep Operator Network with branch and continuous trunk nets", "Yes")
    table.add_row("gno", "Graph Neural Operator for irregular meshes and point clouds", "Yes")
    table.add_row("hybrid", "Multi-scale Fourier + local self-attention hybrid operator", "Yes")
    console.print(table)


@app.command(name="list-pdes")
def list_pdes() -> None:
    """List available PDE problem benchmarks."""
    table = Table(title="Available PDE Systems")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Formulation", style="green")
    table.add_row("heat", "2D Heat diffusion equation", "Spectral Fourier exact")
    table.add_row("navier_stokes", "2D Incompressible Navier-Stokes equations", "Vorticity-stream pseudospectral")
    table.add_row("wave", "2D Acoustic wave equation", "Second-order spectral")
    table.add_row("shallow_water", "2D 1.5-layer shallow water equations", "Nonlinear height & momentum")
    table.add_row("elasticity", "2D Linear elastostatics", "Displacement & stress divergence")
    table.add_row("reaction_diffusion", "2D Gray-Scott reaction diffusion system", "Two-species coupled PDE")
    console.print(table)


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to YAML experiment config"),
    device: str = typer.Option("auto", help="Device: 'cpu', 'cuda', 'cuda:0', 'auto'"),
    epochs: Optional[int] = typer.Option(None, help="Override number of epochs"),
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Train a neural operator from a YAML config file."""
    _setup_logging(verbose)

    import torch
    from torch.utils.data import DataLoader

    from operatorlab.configs.schema import from_yaml
    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.schedulers import build_scheduler
    from operatorlab.training.trainer import Trainer

    # Load config
    config = from_yaml(config_path)
    if epochs is not None:
        config.training.epochs = epochs

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold green]Device:[/] {device}")
    console.print(f"[bold green]Config:[/] {config_path}")

    # Build PDE and generate data
    pde = _build_pde(config.pde)
    console.print(f"[bold green]PDE:[/] {pde.name}")

    train_data = pde.generate_dataset(
        n_samples=config.data.n_train,
        resolution=config.pde.resolution,
        dt=config.pde.dt,
        T=config.pde.T,
        seed=config.data.seed,
    )
    val_data = pde.generate_dataset(
        n_samples=config.data.n_val,
        resolution=config.pde.resolution,
        dt=config.pde.dt,
        T=config.pde.T,
        seed=config.data.seed + 1,
    )

    train_ds = InMemoryDataset(train_data["a"], train_data["u"], normalize=True)
    val_ds = InMemoryDataset(val_data["a"], val_data["u"], normalize=True)

    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Build model
    model = _build_model(config.model)
    n_params = model.count_parameters()
    console.print(f"[bold green]Model:[/] {config.model.type} ({n_params:,} parameters)")

    # Build optimizer
    if config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    # Build scheduler
    scheduler = build_scheduler(
        optimizer, config.training.scheduler, config.training.epochs,
    )

    # Build loss
    loss_fn = OperatorLoss(
        l2_weight=config.training.loss.l2_weight,
        h1_weight=config.training.loss.h1_weight,
        pde_weight=config.training.loss.pde_weight,
        pde=pde if config.training.loss.pde_weight > 0 else None,
    )

    # Resume from checkpoint if requested
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        console.print(f"[bold yellow]Resumed from:[/] {resume} (epoch {ckpt['epoch']})")

    # Build trainer and run
    checkpoint_dir = Path(config.experiment.output_dir) / config.experiment.name / "checkpoints"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_amp=config.training.amp,
        grad_clip=config.training.grad_clip,
        checkpoint_dir=checkpoint_dir,
        config=config.to_dict(),
    )

    console.print(f"[bold cyan]Training for {config.training.epochs} epochs...[/]")
    result = trainer.train(train_loader, val_loader, config.training.epochs)

    # Print results
    console.print(f"\n[bold green]Training complete![/]")
    console.print(f"  Best val loss: {result.best_val_loss:.6f} (epoch {result.best_epoch})")
    console.print(f"  Final train loss: {result.final_train_loss:.6f}")
    console.print(f"  Total time: {result.total_time:.1f}s")
    console.print(f"  Checkpoints: {checkpoint_dir}")


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML"),
    pde: Optional[str] = typer.Option(None, "--pde", help="Override PDE type (e.g. heat, navier_stokes, wave)"),
    resolution: Optional[int] = typer.Option(None, help="Override test resolution"),
    resolution_sweep: bool = typer.Option(False, "--resolution-sweep", help="Run resolution sweep"),
    physics_violation: bool = typer.Option(False, "--physics-violation", help="Run physics violation audit"),
    n_samples: int = typer.Option(200, help="Number of test samples"),
    device: str = typer.Option("auto", help="Device"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a trained checkpoint."""
    _setup_logging(verbose)

    import torch
    from torch.utils.data import DataLoader
    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.training.losses import OperatorLoss
    from operatorlab.training.trainer import Trainer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    model_config_dict = ckpt_config.get("model", {})

    # Rebuild model from checkpoint config
    from operatorlab.configs.schema import ModelConfig, PDEConfig, _merge_dataclass

    model_cfg = _merge_dataclass(ModelConfig, model_config_dict) if model_config_dict else ModelConfig()
    model = _build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    console.print(f"[bold green]Loaded checkpoint:[/] {checkpoint}")
    console.print(f"  Model: {model_cfg.type} ({model.count_parameters():,} params)")

    # Build PDE for test data
    pde_config_dict = ckpt_config.get("pde", {})
    pde_cfg = _merge_dataclass(PDEConfig, pde_config_dict) if pde_config_dict else PDEConfig()
    if pde is not None:
        pde_cfg.type = pde
    pde_problem = _build_pde(pde_cfg)

    test_res = resolution or pde_cfg.resolution
    test_data = pde_problem.generate_dataset(
        n_samples=n_samples,
        resolution=test_res,
        dt=pde_cfg.dt,
        T=pde_cfg.T,
        seed=9999,
    )
    test_ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=True)
    test_loader = DataLoader(test_ds, batch_size=20, shuffle=False)

    loss_fn = OperatorLoss()
    trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters()),
                      loss_fn=loss_fn, device=device, use_amp=False)
    result = trainer.evaluate(test_loader, resolution=test_res)

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Resolution", f"{test_res}×{test_res}")
    table.add_row("Relative L2", f"{result.l2_error:.6f}")
    table.add_row("Relative H1", f"{result.h1_error:.6f}")
    table.add_row("Loss", f"{result.loss:.6f}")
    table.add_row("N Samples", str(result.n_samples))
    console.print(table)

    if physics_violation:
        from operatorlab.evaluation.physics_violation import analyze_physics_violations
        report = analyze_physics_violations(model, pde_problem, test_loader, device=device)
        pv_table = Table(title="Physics Violation Audit")
        pv_table.add_column("Metric", style="cyan")
        pv_table.add_column("Value", style="green")
        pv_table.add_row("Mean Residual Norm", f"{report.mean_residual_norm:.6f}")
        pv_table.add_row("Max Residual Norm", f"{report.max_residual_norm:.6f}")
        pv_table.add_row("Energy Spectrum Error", f"{report.energy_spectrum_error:.6f}")
        pv_table.add_row("Divergence Norm", f"{report.divergence_norm:.6f}")
        for k, v in report.conservation_violations.items():
            pv_table.add_row(f"Conservation ({k})", f"{v:.6f}")
        console.print(pv_table)

    if resolution_sweep:
        from operatorlab.evaluation.resolution_sweep import resolution_sweep as run_sweep
        sweep_results = run_sweep(
            model=model, pde=pde_problem,
            base_resolution=pde_cfg.resolution,
            target_resolutions=[32, 64, 128, 256],
            n_test_samples=min(n_samples, 50),
            device=device,
        )
        sweep_table = Table(title="Resolution Sweep")
        sweep_table.add_column("Resolution", style="cyan")
        sweep_table.add_column("L2 Error", style="green")
        for r in sweep_results.results:
            sweep_table.add_row(f"{r.resolution}×{r.resolution}", f"{r.l2_error:.6f}")
        console.print(sweep_table)


@app.command()
def compare(
    checkpoints: list[Path] = typer.Argument(..., help="Checkpoint files to compare"),
    label_a: Optional[str] = typer.Option(None, "--label-a", help="Label for first checkpoint"),
    label_b: Optional[str] = typer.Option(None, "--label-b", help="Label for second checkpoint"),
    resolution_sweep: bool = typer.Option(False, "--resolution-sweep", help="Compare resolution sweeps"),
    save_plot: Optional[Path] = typer.Option(None, "--save-plot", help="Save comparison plot to file"),
    n_samples: int = typer.Option(200, help="Number of test samples"),
    device: str = typer.Option("auto", help="Device"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare multiple trained checkpoints."""
    _setup_logging(verbose)

    import torch
    from torch.utils.data import DataLoader
    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.training.losses import OperatorLoss, relative_l2
    from operatorlab.training.trainer import Trainer
    from operatorlab.configs.schema import ModelConfig, PDEConfig, _merge_dataclass

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    table = Table(title="Model Comparison")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Params", style="white")
    table.add_column("L2 Error", style="green")
    table.add_column("H1 Error", style="magenta")
    table.add_column("Loss", style="yellow")

    loaded_models: list[tuple[str, torch.nn.Module, object, list[float]]] = []

    for i, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {})

        model_cfg = _merge_dataclass(ModelConfig, ckpt_config.get("model", {}))
        model = _build_model(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        pde_cfg = _merge_dataclass(PDEConfig, ckpt_config.get("pde", {}))
        pde = _build_pde(pde_cfg)

        test_data = pde.generate_dataset(
            n_samples=n_samples, resolution=pde_cfg.resolution,
            dt=pde_cfg.dt, T=pde_cfg.T, seed=9999,
        )
        test_ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=True)
        test_loader = DataLoader(test_ds, batch_size=20, shuffle=False)

        loss_fn = OperatorLoss()
        trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters()),
                          loss_fn=loss_fn, device=device, use_amp=False)
        result = trainer.evaluate(test_loader)

        # Compute per-sample L2 error for statistical tests
        per_sample_errors: list[float] = []
        with torch.no_grad():
            for a, u, grid in test_loader:
                a, u, grid = a.to(device), u.to(device), grid.to(device)
                if grid.ndim == 3:
                    grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)
                pred = model(a, grid)
                for s in range(a.shape[0]):
                    per_sample_errors.append(relative_l2(pred[s:s+1], u[s:s+1]).item())

        name = ckpt_path.name
        if i == 0 and label_a:
            name = label_a
        elif i == 1 and label_b:
            name = label_b

        loaded_models.append((name, model, pde, per_sample_errors))

        table.add_row(
            name,
            model_cfg.type,
            f"{model.count_parameters():,}",
            f"{result.l2_error:.6f}",
            f"{result.h1_error:.6f}",
            f"{result.loss:.6f}",
        )

    console.print(table)

    # Statistical significance test if exactly two models
    if len(loaded_models) == 2:
        try:
            from scipy import stats
            name_a, _, _, errs_a = loaded_models[0]
            name_b, _, _, errs_b = loaded_models[1]
            diff = [a - b for a, b in zip(errs_a, errs_b)]
            stat_res = stats.wilcoxon(diff)
            p_val = float(stat_res.pvalue)
            sig = "[bold green]Statistically Significant (p < 0.05)[/]" if p_val < 0.05 else "[yellow]Not Statistically Significant (p >= 0.05)[/]"
            console.print(f"\n[bold cyan]Wilcoxon Signed-Rank Test ({name_a} vs {name_b}):[/]")
            console.print(f"  p-value: {p_val:.4e} — {sig}")
        except Exception as exc:
            logger.debug("Statistical test skipped: %s", exc)

    if resolution_sweep and len(loaded_models) > 0:
        from operatorlab.evaluation.resolution_sweep import resolution_sweep as run_sweep
        sweep_data: dict[str, list[float]] = {}
        resolutions = [32, 64, 128, 256]

        sweep_table = Table(title="Resolution Scaling Comparison")
        sweep_table.add_column("Resolution", style="cyan")
        for name, _, _, _ in loaded_models:
            sweep_table.add_column(name, style="green")

        for res in resolutions:
            row = [f"{res}×{res}"]
            for name, mod, pde_prob, _ in loaded_models:
                sweep = run_sweep(
                    model=mod, pde=pde_prob,
                    base_resolution=64,
                    target_resolutions=[res],
                    n_test_samples=min(n_samples, 50),
                    device=device,
                )
                err = sweep.results[0].l2_error
                row.append(f"{err:.6f}")
                sweep_data.setdefault(name, []).append(err)
            sweep_table.add_row(*row)

        console.print(sweep_table)

        if save_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for name, errs in sweep_data.items():
                ax.plot(resolutions, errs, marker="o", label=name)
            ax.set_xlabel("Resolution")
            ax.set_ylabel("Relative $L_2$ Error")
            ax.set_title("Zero-Shot Resolution Scaling Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_plot = Path(save_plot)
            save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            console.print(f"[bold green]Saved resolution scaling plot to:[/] {save_plot}")


@app.command()
def visualize(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    mode: str = typer.Option("field", help="Visualization mode: field, spectrum, trajectory"),
    sample_idx: int = typer.Option(0, help="Sample index to visualize"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    device: str = typer.Option("auto", help="Device"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate visualizations from a trained checkpoint."""
    _setup_logging(verbose)
    console.print(f"[bold cyan]Generating {mode} visualization...[/]")

    import torch
    from operatorlab.configs.schema import ModelConfig, PDEConfig, _merge_dataclass

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    model_cfg = _merge_dataclass(ModelConfig, ckpt_config.get("model", {}))
    model = _build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    pde_cfg = _merge_dataclass(PDEConfig, ckpt_config.get("pde", {}))
    pde = _build_pde(pde_cfg)

    test_data = pde.generate_dataset(
        n_samples=max(sample_idx + 1, 10), resolution=pde_cfg.resolution,
        dt=pde_cfg.dt, T=pde_cfg.T, seed=12345,
    )

    save_path = output or Path(f"visualization_{mode}.png")

    if mode == "field":
        from operatorlab.visualization.fields import plot_field_comparison
        a_sample = test_data["a"][sample_idx:sample_idx+1].to(device)
        u_sample = test_data["u"][sample_idx:sample_idx+1].to(device)
        with torch.no_grad():
            pred = model(a_sample)
        plot_field_comparison(a_sample, pred, u_sample, save_path=save_path)
    elif mode == "spectrum":
        from operatorlab.visualization.spectra import plot_error_spectrum
        a_sample = test_data["a"][sample_idx:sample_idx+1].to(device)
        u_sample = test_data["u"][sample_idx:sample_idx+1].to(device)
        with torch.no_grad():
            pred = model(a_sample)
        plot_error_spectrum(pred, u_sample, save_path=save_path)
    elif mode == "trajectory":
        from operatorlab.visualization.trajectories import animate_rollout
        a_sample = test_data["a"][sample_idx:sample_idx+1].to(device)
        save_path = output or Path("rollout.gif")
        animate_rollout(model, pde, a_sample, T=pde_cfg.T, n_steps=10, save_path=save_path)
    else:
        console.print(f"[red]Unknown mode: {mode}[/]")
        raise typer.Exit(1)

    console.print(f"[bold green]Saved to:[/] {save_path}")


@app.command()
def generate(
    pde_type: str = typer.Argument(..., help="PDE type: heat, navier_stokes, wave, etc."),
    resolution: int = typer.Option(64, help="Spatial resolution"),
    n_samples: int = typer.Option(1000, help="Number of samples to generate"),
    output: Path = typer.Option(Path("data/"), "--output", "-o", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate PDE training data."""
    _setup_logging(verbose)

    from operatorlab.configs.schema import PDEConfig

    pde_config = PDEConfig(type=pde_type, resolution=resolution)
    pde = _build_pde(pde_config)

    console.print(f"[bold cyan]Generating {n_samples} {pde_type} samples at {resolution}×{resolution}...[/]")

    import h5py
    data = pde.generate_dataset(
        n_samples=n_samples,
        resolution=resolution,
        dt=pde_config.dt,
        T=pde_config.T,
        seed=seed,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    filename = f"{pde_type}_{resolution}_{n_samples}_{seed}.h5"
    filepath = output / filename

    with h5py.File(filepath, "w") as f:
        f.create_dataset("a", data=data["a"].numpy())
        f.create_dataset("u", data=data["u"].numpy())

    console.print(f"[bold green]Saved {n_samples} samples to:[/] {filepath}")
    console.print(f"  a shape: {tuple(data['a'].shape)}")
    console.print(f"  u shape: {tuple(data['u'].shape)}")


def _load_model_and_pde_from_checkpoint(checkpoint: Path, device: str, override_pde: Optional[str] = None):
    """Load model and PDE problem instance from a saved checkpoint."""
    import torch
    from operatorlab.configs.schema import ModelConfig, PDEConfig, _merge_dataclass

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    model_config_dict = ckpt_config.get("model", {})
    model_cfg = _merge_dataclass(ModelConfig, model_config_dict) if model_config_dict else ModelConfig()
    model = _build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    pde_config_dict = ckpt_config.get("pde", {})
    pde_cfg = _merge_dataclass(PDEConfig, pde_config_dict) if pde_config_dict else PDEConfig()
    if override_pde is not None:
        pde_cfg.type = override_pde
    pde_problem = _build_pde(pde_cfg)

    return model, pde_problem, model_cfg, pde_cfg


@app.command()
def ood(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    pde: Optional[str] = typer.Option(None, "--pde", help="Override PDE type (e.g. navier_stokes, heat)"),
    resolution: Optional[int] = typer.Option(None, help="Base resolution"),
    n_samples: int = typer.Option(30, help="Samples per OOD condition"),
    device: str = typer.Option("auto", help="Device ('cpu', 'cuda', 'auto')"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Out-of-Distribution (OOD) operator generalization benchmark."""
    _setup_logging(verbose)
    import json
    import torch
    from operatorlab.evaluation.ood import run_ood_generalization_benchmark

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold cyan]Loading checkpoint for OOD evaluation:[/] {checkpoint}")
    model, pde_problem, model_cfg, pde_cfg = _load_model_and_pde_from_checkpoint(checkpoint, device, override_pde=pde)

    base_res = resolution or pde_cfg.resolution
    console.print(f"  Model: [bold]{model_cfg.type}[/] | PDE: [bold]{pde_problem.name}[/] | Base Res: {base_res}×{base_res}")
    console.print("[bold yellow]Running OOD Generalization Suite...[/]")

    report = run_ood_generalization_benchmark(
        model=model,
        pde=pde_problem,
        base_resolution=base_res,
        n_samples=n_samples,
        device=device,
    )

    console.print("\n" + report.summary_table())

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"\n[bold green]Saved OOD report to:[/] {output}")


@app.command()
def stress(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    pde: Optional[str] = typer.Option(None, "--pde", help="Override PDE type"),
    resolution: Optional[int] = typer.Option(None, help="Test resolution"),
    n_samples: int = typer.Option(50, help="Number of test samples"),
    device: str = typer.Option("auto", help="Device ('cpu', 'cuda', 'auto')"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run operator robustness and stress testing under perturbations."""
    _setup_logging(verbose)
    import json
    import torch
    from torch.utils.data import DataLoader
    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.evaluation.stress import run_stress_test

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold cyan]Loading checkpoint for stress testing:[/] {checkpoint}")
    model, pde_problem, model_cfg, pde_cfg = _load_model_and_pde_from_checkpoint(checkpoint, device, override_pde=pde)

    test_res = resolution or pde_cfg.resolution
    console.print(f"  Model: [bold]{model_cfg.type}[/] | PDE: [bold]{pde_problem.name}[/] | Resolution: {test_res}×{test_res}")
    console.print(f"Generating {n_samples} evaluation samples...")

    test_data = pde_problem.generate_dataset(n_samples=n_samples, resolution=test_res, seed=777)
    ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=False)
    loader = DataLoader(ds, batch_size=min(20, n_samples))

    console.print("[bold yellow]Executing perturbation stress suite...[/]")
    report = run_stress_test(model=model, test_loader=loader, device=device)

    console.print("\n" + report.summary_table())

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"\n[bold green]Saved robustness report to:[/] {output}")


@app.command()
def audit(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    pde: Optional[str] = typer.Option(None, "--pde", help="Override PDE type"),
    resolution: Optional[int] = typer.Option(None, help="Test resolution"),
    n_samples: int = typer.Option(50, help="Number of test samples"),
    max_steps: int = typer.Option(25, help="Max autoregressive rollout steps"),
    device: str = typer.Option("auto", help="Device ('cpu', 'cuda', 'auto')"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save audit to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Audit scientific validity and physical invariants of a trained model."""
    _setup_logging(verbose)
    import json
    import torch
    from torch.utils.data import DataLoader
    from operatorlab.data.datasets import InMemoryDataset
    from operatorlab.evaluation.scientific_audit import run_scientific_audit

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold cyan]Loading checkpoint for scientific audit:[/] {checkpoint}")
    model, pde_problem, model_cfg, pde_cfg = _load_model_and_pde_from_checkpoint(checkpoint, device, override_pde=pde)

    test_res = resolution or pde_cfg.resolution
    console.print(f"  Model: [bold]{model_cfg.type}[/] | PDE: [bold]{pde_problem.name}[/] | Resolution: {test_res}×{test_res}")
    console.print(f"Generating {n_samples} audit samples...")

    test_data = pde_problem.generate_dataset(n_samples=n_samples, resolution=test_res, seed=888)
    ds = InMemoryDataset(test_data["a"], test_data["u"], normalize=False)
    loader = DataLoader(ds, batch_size=min(20, n_samples))

    console.print("[bold yellow]Auditing physical invariants and conservation laws...[/]")
    card = run_scientific_audit(model=model, pde=pde_problem, test_loader=loader, device=device, max_rollout_steps=max_steps)

    console.print("\n" + card.summary_card())

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(card.to_dict(), f, indent=2)
        console.print(f"\n[bold green]Saved scientific audit to:[/] {output}")


@app.command()
def arena(
    pde: str = typer.Option("navier_stokes", "--pde", help="Target PDE for competition"),
    resolution: int = typer.Option(64, "--resolution", "-r", help="Grid resolution"),
    n_samples: int = typer.Option(25, "--samples", "-n", help="Samples per condition"),
    device: str = typer.Option("auto", help="Device ('cpu', 'cuda', 'auto')"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save leaderboard to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run OperatorArena multi-model competitive benchmark and leaderboard."""
    _setup_logging(verbose)
    import torch
    from operatorlab.arena.benchmark import ArenaBenchmark
    from operatorlab.configs.schema import PDEConfig
    from operatorlab.models.deeponet import DeepONet
    from operatorlab.models.fno import FNO2d
    from operatorlab.models.hybrid import HybridOperator
    from operatorlab.models.tfno import TFNO2d

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pde_problem = _build_pde(PDEConfig(type=pde, resolution=resolution))
    console.print(f"[bold cyan]Starting OperatorArena on {pde_problem.name} ({resolution}×{resolution})...[/]")

    modes = min(12, resolution // 2)
    models = {
        "FNO": FNO2d(modes1=modes, modes2=modes, width=32, n_layers=4),
        "TFNO": TFNO2d(modes1=modes, modes2=modes, width=32, n_layers=4, rank=8),
        "DeepONet": DeepONet(branch_input_dim=resolution * resolution, hidden_dim=64, n_basis=64),
        "Hybrid": HybridOperator(modes1=modes, modes2=modes, width=32, n_fno_layers=2, n_attention_layers=1),
    }

    arena_runner = ArenaBenchmark(pde=pde_problem, resolution=resolution, n_eval_samples=n_samples, device=device)
    leaderboard = arena_runner.run(models)

    console.print("\n" + leaderboard.terminal_table())

    if output is not None:
        leaderboard.save_json(output)
        console.print(f"\n[bold green]Saved Arena leaderboard to:[/] {output}")


if __name__ == "__main__":
    app()

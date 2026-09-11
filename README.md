# OperatorLab

Research-grade neural operator framework for learning mappings between function spaces over physical domains.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Generate training data
operatorlab generate heat --resolution 64 --n-samples 1000 --output data/

# Train a model
operatorlab train configs/heat_equation.yaml

# Evaluate with resolution sweep
operatorlab evaluate checkpoint.pt --resolution-sweep

# Compare architectures
operatorlab compare checkpoint_a.pt checkpoint_b.pt --resolution-sweep
```

## Architecture

OperatorLab implements five neural operator architectures:

- **FNO** — Fourier Neural Operator (spectral convolution in Fourier domain)
- **TFNO** — Tensorized FNO (Tucker-decomposed spectral weights)
- **DeepONet** — Deep Operator Network (branch-trunk architecture)
- **GNO** — Graph Neural Operator (message-passing on spatial graphs)
- **Hybrid** — Fourier + local attention

All architectures support **zero-shot resolution transfer**: train at one resolution, evaluate at any other.

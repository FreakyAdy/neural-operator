# Contributing to OperatorLab

Thank you for your interest in contributing to **OperatorLab**! We welcome contributions from researchers, engineers, and scientific machine learning practitioners.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/FreakyAdy/neural-operator.git
   cd neural-operator
   ```
3. **Set up a virtual environment and install development dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Running Tests
Make sure all tests pass before submitting a Pull Request:
```bash
pytest tests/ -v
```

### Adding New Neural Operators
All models inherit from `torch.nn.Module` and should adhere to the following conventions:
1. Accept input tensors with coordinates/grid channel concatenation where applicable.
2. Support arbitrary spatial resolution at inference (zero-shot transfer capability).
3. Place model architecture in `operatorlab/models/`.
4. Register the new model in `operatorlab/models/__init__.py` and update `operatorlab/cli.py:list_models`.
5. Add unit tests covering parameter initialization, forward pass shape invariance, and backward gradient flow under `tests/`.

### Adding New Physical Systems
1. Place PDE solver / simulator in `operatorlab/physics/`.
2. Implement exact or high-order numerical scheme (e.g. pseudo-spectral, finite-difference).
3. Provide physics residual calculation function for physics-informed evaluation.
4. Add unit tests under `tests/test_physics.py`.

## Submitting Pull Requests
- Ensure PR descriptions clearly state the motivation, architectural changes, and validation results.
- Include zero-shot resolution transfer metrics if introducing a new operator architecture.

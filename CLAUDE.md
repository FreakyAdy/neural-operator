# CLAUDE.md — OperatorLab

> This file is the source of truth for Claude when working on this codebase.
> Read it fully before writing any code, suggesting any architecture, or modifying any file.

---

## Project identity

**OperatorLab** is a research-grade, open-source neural operator framework for learning mappings between function spaces over physical domains.

It is not a tutorial project. It is not a demo. It is built to produce publishable benchmarks.

The core idea: instead of learning `input tensor → output tensor`, we learn **operators**: mappings from one function to another, where both live in infinite-dimensional function spaces. A trained neural operator takes boundary/initial conditions as input and outputs the entire physical field — and generalizes to resolutions it has never seen during training.

---

## What this project is NOT

- Not a clone of `neuraloperator` (the existing PyTorch library). Do not import from it.
- Not a physics simulator. We do not solve PDEs numerically (except to generate training data).
- Not a demo notebook collection. Everything must be usable from the CLI.
- Not over-engineered. Avoid abstract base class pyramids, metaclass tricks, or plugin registries unless they solve a real problem.

---

## Repository layout

```
operatorlab/
├── CLAUDE.md                        ← this file
├── README.md
├── pyproject.toml
├── setup.cfg
│
├── operatorlab/                     ← main Python package
│   ├── __init__.py
│   ├── cli.py                       ← `operatorlab` entry-point (click/typer)
│   │
│   ├── models/                      ← neural operator architectures
│   │   ├── __init__.py
│   │   ├── base.py                  ← NeuralOperator ABC
│   │   ├── fno.py                   ← Fourier Neural Operator
│   │   ├── tfno.py                  ← Tensorized FNO
│   │   ├── deeponet.py              ← Deep Operator Network
│   │   ├── gno.py                   ← Graph Neural Operator
│   │   └── hybrid.py                ← Fourier + attention hybrid
│   │
│   ├── physics/                     ← PDE problem definitions
│   │   ├── __init__.py
│   │   ├── base.py                  ← PDE ABC
│   │   ├── navier_stokes.py
│   │   ├── heat.py
│   │   ├── wave.py
│   │   ├── shallow_water.py
│   │   ├── elasticity.py
│   │   └── reaction_diffusion.py
│   │
│   ├── data/                        ← dataset generation + loaders
│   │   ├── __init__.py
│   │   ├── generators/              ← numerical solvers (FDM/spectral)
│   │   │   ├── spectral_ns.py       ← spectral Navier-Stokes solver
│   │   │   └── ...
│   │   ├── datasets.py              ← PyTorch Dataset wrappers
│   │   └── transforms.py           ← normalization, mesh encoding
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               ← main Trainer class
│   │   ├── losses.py                ← relative L2, H1, PDE residual
│   │   ├── schedulers.py
│   │   └── distributed.py          ← DDP helpers
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               ← L2, H1, energy spectrum error
│   │   ├── resolution_sweep.py      ← zero-shot resolution scaling
│   │   └── physics_violation.py    ← divergence, conservation checks
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── fields.py                ← field plots, vorticity, streamlines
│   │   ├── spectra.py               ← error spectrum visualization
│   │   └── trajectories.py         ← animated temporal rollouts
│   │
│   ├── registry/
│   │   ├── __init__.py
│   │   └── experiment.py           ← experiment tracking (no MLflow dep required)
│   │
│   └── configs/                     ← config schema (dataclasses or Pydantic)
│       ├── __init__.py
│       └── schema.py
│
├── configs/                         ← YAML experiment configs
│   ├── navier_stokes.yaml
│   ├── heat_equation.yaml
│   ├── wave_equation.yaml
│   └── shallow_water.yaml
│
├── benchmarks/                      ← reproducible benchmark scripts
│   ├── resolution_generalization.py
│   ├── architecture_comparison.py
│   └── physics_violation_audit.py
│
├── tests/
│   ├── unit/
│   │   ├── test_fno.py
│   │   ├── test_deeponet.py
│   │   ├── test_losses.py
│   │   └── test_resolution_transfer.py
│   └── integration/
│       ├── test_train_smoke.py
│       └── test_cli.py
│
└── notebooks/                       ← optional, for demos only
    └── resolution_scaling_demo.ipynb
```

---

## Core concepts — understand these before writing code

### Function space learning

A neural operator `G : A → U` maps between infinite-dimensional function spaces:
- `A`: input function space (e.g., initial vorticity field `a(x)`)
- `U`: output function space (e.g., vorticity at time `T`, `u(x, T)`)

The key property: `G` is **discretization-independent**. It does not depend on the resolution of the input grid. This is what makes zero-shot resolution transfer possible.

### Resolution transfer

- Train on `64×64` grid
- Evaluate (zero-shot) on `128×128`, `256×256`, `512×512`
- The model never retrains; only the input grid changes
- FNO achieves this via spectral truncation: the Fourier modes are fixed, the spatial grid can vary
- All architectures must support this — do not hardcode grid size anywhere

### The integral kernel formulation

The FNO applies this update at each layer:

```
v_{t+1}(x) = σ(W·v_t(x) + K(a; φ)·v_t(x))
```

Where `K(a; φ)` is a kernel integral operator:

```
(K·v)(x) = ∫ κ(x, y, a(x), a(y); φ) · v(y) dy
```

In the Fourier domain this becomes pointwise multiplication:

```
K̂(v̂)(k) = R(k) · v̂(k)   for |k| ≤ k_max
```

Where `R(k)` are learnable complex-valued weight matrices per frequency mode.

---

## Architecture specifications

### 1. FNO (Fourier Neural Operator)

**File:** `operatorlab/models/fno.py`

**Core components:**

```python
class SpectralConv2d(nn.Module):
    """
    Fourier layer: FFT → pointwise multiply by learnable R(k) → iFFT.
    Operates on (batch, channels, height, width).
    """

class FNO2d(NeuralOperator):
    """
    Stack of L Fourier layers with residual skip connections.
    
    Args:
        modes1, modes2: number of Fourier modes in each spatial dim
        width: channel width in the lifted space
        n_layers: number of Fourier integral operator layers
        input_dim: channels of input (usually 1 + 2 for field + grid coords)
        output_dim: channels of output
    """
```

**Implementation notes:**
- Input is always lifted: `v = P(a)` where `P` is a pointwise MLP
- Output is projected back: `u = Q(v_L)` where `Q` is a pointwise MLP
- Grid coordinates `(x, y) ∈ [0,1]²` are always appended as input channels
- Spectral truncation at `k_max` modes enables resolution generalization
- Use `torch.fft.rfft2` and `torch.fft.irfft2` (not the deprecated `torch.rfft`)
- Weight matrices are complex: `nn.Parameter(torch.randn(..., dtype=torch.cfloat))`

### 2. TFNO (Tensorized FNO)

**File:** `operatorlab/models/tfno.py`

TFNO factorizes the spectral weights using tensor decomposition (Tucker or CP):

```python
class TuckerSpectralConv2d(SpectralConv2d):
    """
    Replaces dense R(k) with Tucker-decomposed factors.
    Reduces parameter count by ~10x at minimal accuracy cost.
    Use tensorly for decomposition if available, else implement Tucker manually.
    """
```

**Key difference from FNO:** The spectral weight tensor `R` of shape `(in_ch, out_ch, modes1, modes2)` is replaced by Tucker factors. This allows training larger models with the same VRAM budget.

### 3. DeepONet (Deep Operator Network)

**File:** `operatorlab/models/deeponet.py`

Two-branch architecture:

```python
class DeepONet(NeuralOperator):
    """
    Branch net: encodes the input function a evaluated at sensor points.
    Trunk net: encodes the query point x where we want to evaluate u(x).
    Output: dot product of branch and trunk outputs.
    
    G(a)(x) = Σ_k branch_k(a) · trunk_k(x) + b
    """
```

**Implementation notes:**
- Branch net input: `a` evaluated at `m` fixed sensor points → shape `(batch, m)`
- Trunk net input: query coordinate `x ∈ R^d` → shape `(batch, n_queries, d)`
- Works naturally on irregular meshes (trunk net is pointwise)
- Pod-DeepONet variant: use POD basis functions as trunk output

### 4. GNO (Graph Neural Operator)

**File:** `operatorlab/models/gno.py`

For **irregular meshes** and **unstructured grids**:

```python
class GNO(NeuralOperator):
    """
    Message-passing neural network over the graph of spatial points.
    Kernel integration approximated by neighbor aggregation.
    Uses torch_geometric if available, else custom sparse message passing.
    """
```

**Notes:**
- Build graph from point cloud using radius-based or kNN neighborhoods
- Edge features: relative position vector `x_i - x_j`, optionally `|x_i - x_j|`
- Each message-passing layer approximates one kernel integral operator layer
- Support both fixed-topology (fast) and dynamic-graph modes

### 5. Fourier-Attention Hybrid

**File:** `operatorlab/models/hybrid.py`

Combines FNO layers with sparse attention:

```python
class HybridOperator(NeuralOperator):
    """
    Alternates between:
    - Fourier layers: capture global, long-range structure
    - Local attention layers: capture fine-grained, local structure
    
    Motivation: pure FNO misses sharp local features;
    pure attention is O(n^2) in spatial resolution.
    """
```

---

## Base class contract

**File:** `operatorlab/models/base.py`

Every model must implement:

```python
class NeuralOperator(nn.Module, ABC):
    @abstractmethod
    def forward(self, a: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            a: input function values, shape (batch, *spatial_dims, in_channels)
            grid: spatial coordinates, shape (batch, *spatial_dims, space_dim)
                  if None, model must construct a uniform grid internally
        Returns:
            u: output function values, shape (batch, *spatial_dims, out_channels)
        """

    @abstractmethod
    def count_parameters(self) -> int: ...

    def supports_resolution_transfer(self) -> bool:
        """Return True if model is resolution-independent by design."""
        return False

    def get_config(self) -> dict:
        """Return hyperparameter dict for experiment registry."""
        ...
```

---

## Physics module specifications

**File:** `operatorlab/physics/base.py`

```python
class PDEProblem(ABC):
    name: str
    spatial_dim: int
    
    @abstractmethod
    def generate_dataset(
        self,
        n_samples: int,
        resolution: int,
        dt: float,
        T: float,
        seed: int = 42,
    ) -> dict[str, Tensor]:
        """
        Returns dict with keys: 'a' (initial condition), 'u' (target field).
        Both tensors of shape (n_samples, *spatial_grid, channels).
        """
    
    @abstractmethod
    def compute_residual(self, u: Tensor, t: float) -> Tensor:
        """
        Compute PDE residual for physics-informed loss.
        Lower is better; used as a diagnostic, not always in training loss.
        """
    
    @abstractmethod
    def check_conservation(self, u: Tensor) -> dict[str, float]:
        """
        Returns dict of conserved quantity violations.
        Example: {'mass': 1e-4, 'energy': 2e-3}
        """
```

### Navier-Stokes 2D (vorticity form)

**File:** `operatorlab/physics/navier_stokes.py`

This is the primary benchmark. Use the **vorticity-stream function formulation**:

```
∂ω/∂t + (u·∇)ω = ν∇²ω + f
∇²ψ = -ω
u = ∂ψ/∂y,  v = -∂ψ/∂x
```

Data generation:
- Use spectral methods (pseudo-spectral) to solve on periodic `[0,1]²` domain
- Forcing: `f(x,y) = 0.1(sin(2π(x+y)) + cos(2π(x+y)))`
- Viscosity: `ν = 1e-3` for turbulent regime, `ν = 1e-4` for harder setting
- Save initial vorticity `ω(x, 0)` as input; vorticity at `T=1` as target
- Precompute and cache to `data/navier_stokes_{resolution}_{n_samples}.h5`

### Heat Equation (2D)

```
∂u/∂t = α∇²u,   u(x, 0) = a(x)
```

Simplest benchmark. Use for unit testing and fast iteration.

### Wave Equation

```
∂²u/∂t² = c²∇²u
```

Tests temporal generalization. Use staggered time steps.

### Reaction-Diffusion

```
∂u/∂t = d_u∇²u + f(u,v)
∂v/∂t = d_v∇²v + g(u,v)
```

Gray-Scott model. Two-component field. Tests multi-channel operator learning.

---

## Loss functions

**File:** `operatorlab/training/losses.py`

### Relative L2 (primary metric)

```python
def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """
    ||u_pred - u_target||_2 / ||u_target||_2
    Computed per sample, then averaged over batch.
    """
```

### H1 (Sobolev) loss

```python
def h1_loss(pred: Tensor, target: Tensor, dx: float) -> Tensor:
    """
    ||u_pred - u_target||_{H^1} = ||u_pred - u_target||_{L^2}
                                 + ||∇u_pred - ∇u_target||_{L^2}
    Finite-difference gradients. Penalizes rough solutions.
    """
```

### PDE residual loss

```python
def pde_residual_loss(
    pred: Tensor,
    pde: PDEProblem,
    t: float,
    weight: float = 0.1,
) -> Tensor:
    """
    Physics-informed regularization term.
    Computes PDE residual of the predicted field.
    Add to data loss: total_loss = data_loss + weight * pde_loss.
    """
```

### Combined loss

```python
class OperatorLoss(nn.Module):
    def __init__(
        self,
        l2_weight: float = 1.0,
        h1_weight: float = 0.0,
        pde_weight: float = 0.0,
        pde: Optional[PDEProblem] = None,
    ): ...
```

---

## Training system

**File:** `operatorlab/training/trainer.py`

```python
class Trainer:
    def __init__(
        self,
        model: NeuralOperator,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[...],
        loss_fn: OperatorLoss,
        device: str,
        use_amp: bool = True,          # automatic mixed precision
        grad_clip: float = 1.0,
        log_every: int = 50,
        checkpoint_dir: Path = Path("checkpoints/"),
        registry: Optional[ExperimentRegistry] = None,
    ):

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> TrainingResult: ...

    def evaluate(
        self,
        loader: DataLoader,
        resolution: Optional[int] = None,    # for resolution sweep
    ) -> EvalResult: ...
```

**AMP usage:**
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    pred = model(a)
    loss = loss_fn(pred, u)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Distributed training:**
- Use `torch.nn.parallel.DistributedDataParallel` (not DataParallel)
- Launch with `torchrun --nproc_per_node=N`
- Add `dist.barrier()` before checkpoint saves
- Only rank 0 writes checkpoints and logs

---

## Evaluation system

**File:** `operatorlab/evaluation/resolution_sweep.py`

This is the **killer feature**. Implement carefully.

```python
def resolution_sweep(
    model: NeuralOperator,
    pde: PDEProblem,
    base_resolution: int,
    target_resolutions: list[int],
    n_test_samples: int = 200,
    device: str = "cuda",
) -> ResolutionSweepResult:
    """
    For each resolution in target_resolutions:
    1. Generate test data at that resolution (or load from cache)
    2. Resize input a to target resolution (bilinear interpolation for fields)
    3. Run model forward pass (NO retraining, NO fine-tuning)
    4. Compute relative L2 error
    5. Return table of resolution → error
    
    For FNO: spectral truncation means this works naturally.
    For DeepONet: trunk net query points just change; works naturally.
    For GNO: rebuild graph at new resolution; works naturally.
    """
```

Expected output format:

```
Resolution  | L2 Error  | H1 Error  | Physics Violation
------------+-----------+-----------+------------------
64×64       | 0.032     | 0.051     | 2.1e-4
128×128     | 0.038     | 0.063     | 3.8e-4
256×256     | 0.049     | 0.082     | 7.2e-4
512×512     | 0.061     | 0.104     | 1.1e-3
```

**File:** `operatorlab/evaluation/physics_violation.py`

```python
def analyze_physics_violations(
    model: NeuralOperator,
    pde: PDEProblem,
    test_loader: DataLoader,
) -> PhysicsViolationReport:
    """
    Computes:
    - PDE residual norm per sample
    - Conservation law violations
    - Divergence (for velocity fields)
    - Energy spectrum comparison (predicted vs ground truth)
    """
```

---

## Visualization system

**File:** `operatorlab/visualization/spectra.py`

```python
def plot_error_spectrum(
    pred: Tensor,
    target: Tensor,
    title: str = "",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Computes 2D FFT of (pred - target), plots error energy per wavenumber.
    Shows where in frequency space the model fails.
    Good models: low error at low wavenumbers, some drift at high wavenumbers.
    Bad models: error accumulates at specific frequency bands (instability).
    """
```

**File:** `operatorlab/visualization/trajectories.py`

```python
def animate_rollout(
    model: NeuralOperator,
    pde: PDEProblem,
    initial_condition: Tensor,
    T: float,
    n_steps: int,
    save_path: Path,
) -> None:
    """
    Autoregressively rolls out the model for multi-step prediction.
    Saves side-by-side: ground truth vs prediction, with error field.
    Output: .gif or .mp4
    """
```

---

## CLI specification

**File:** `operatorlab/cli.py`

Use `typer` (preferred) or `click`.

### `operatorlab train`

```bash
operatorlab train configs/navier_stokes.yaml
operatorlab train configs/navier_stokes.yaml --resume checkpoints/run_42/last.pt
operatorlab train configs/navier_stokes.yaml --device cuda:1 --epochs 200
```

Config YAML structure:

```yaml
# configs/navier_stokes.yaml
experiment:
  name: fno_ns_baseline
  seed: 42
  output_dir: experiments/

model:
  type: fno          # fno | tfno | deeponet | gno | hybrid
  modes: 12
  width: 64
  n_layers: 4

pde:
  type: navier_stokes
  viscosity: 1.0e-3
  T: 1.0
  resolution: 64

data:
  n_train: 1000
  n_val: 200
  n_test: 200
  batch_size: 20
  cache_dir: data/

training:
  epochs: 500
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine
  amp: true
  grad_clip: 1.0
  loss:
    l2_weight: 1.0
    h1_weight: 0.1
    pde_weight: 0.0

evaluation:
  resolution_sweep: [64, 128, 256, 512]
  eval_every: 50
```

### `operatorlab evaluate`

```bash
operatorlab evaluate checkpoint.pt
operatorlab evaluate checkpoint.pt --resolution 256
operatorlab evaluate checkpoint.pt --pde heat --n-samples 500
```

Outputs:
- Relative L2, H1 error on test set
- Physics violation report
- Resolution sweep table (if `--resolution-sweep` flag set)

### `operatorlab compare`

```bash
operatorlab compare checkpoint_a.pt checkpoint_b.pt
operatorlab compare checkpoint_a.pt checkpoint_b.pt --label-a "FNO" --label-b "TFNO"
operatorlab compare *.pt --resolution-sweep
```

Outputs:
- Side-by-side metric table
- Resolution scaling plot (saved as PNG)
- Statistical significance test (Wilcoxon rank-sum on per-sample errors)

### `operatorlab visualize`

```bash
operatorlab visualize checkpoint.pt --mode trajectory
operatorlab visualize checkpoint.pt --mode spectrum
operatorlab visualize checkpoint.pt --mode field --sample-idx 0
```

### `operatorlab generate`

```bash
operatorlab generate navier_stokes --resolution 64 --n-samples 2000 --output data/
operatorlab generate heat --resolution 128 --n-samples 5000
```

---

## Experiment registry

**File:** `operatorlab/registry/experiment.py`

No MLflow, no W&B dependency required (but they can be added as optional backends).

```python
class ExperimentRegistry:
    """
    Lightweight JSON-based experiment tracker.
    Stores: config, metrics per epoch, final eval, git hash, timestamp.
    Lives at experiments/{run_name}/
    """
    
    def log_config(self, config: dict) -> None: ...
    def log_metrics(self, epoch: int, metrics: dict) -> None: ...
    def log_eval(self, eval_result: EvalResult) -> None: ...
    def summary(self) -> pd.DataFrame: ...  # compare all runs
```

Each run produces:
```
experiments/fno_ns_baseline_20250912_143022/
├── config.yaml           ← full resolved config
├── metrics.jsonl         ← one JSON line per epoch
├── eval.json             ← final evaluation results
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── plots/
    ├── training_curve.png
    ├── resolution_sweep.png
    └── error_spectrum.png
```

---

## Config schema

**File:** `operatorlab/configs/schema.py`

Use Python dataclasses (not Pydantic, to minimize dependencies):

```python
@dataclass
class ModelConfig:
    type: str
    modes: int = 12
    width: int = 64
    n_layers: int = 4
    input_dim: int = 1
    output_dim: int = 1

@dataclass
class TrainingConfig:
    epochs: int = 500
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    amp: bool = True
    grad_clip: float = 1.0

@dataclass
class ExperimentConfig:
    model: ModelConfig
    pde: PDEConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvalConfig
    experiment: ExperimentMeta
```

---

## Data pipeline

**File:** `operatorlab/data/datasets.py`

```python
class PDEDataset(Dataset):
    """
    Wraps pre-generated PDE data (HDF5 or numpy).
    Supports on-the-fly resolution resampling for multi-resolution training.
    
    Returns:
        a: initial/boundary condition, shape (*spatial, in_ch)
        u: target field, shape (*spatial, out_ch)
        grid: coordinate grid, shape (*spatial, spatial_dim)
    """
    
    def __init__(
        self,
        path: Path,
        split: str = "train",
        resolution: Optional[int] = None,   # if set, resample to this res
        normalize: bool = True,
        device: str = "cpu",
    ): ...
```

**Normalization:**
- Normalize `a` and `u` to zero mean, unit variance (computed on train split only)
- Store normalization stats in dataset file so they can be applied at inference

**Multi-resolution training strategy:**
- Optionally load batches at varying resolutions within a single epoch
- This can further improve resolution generalization

---

## Testing requirements

Every component needs a test. Tests must be fast (< 5 seconds each).

### Unit tests

```python
# tests/unit/test_fno.py
def test_fno_output_shape():
    model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
    a = torch.randn(4, 64, 64, 1)
    out = model(a)
    assert out.shape == (4, 64, 64, 1)

def test_fno_resolution_transfer():
    model = FNO2d(modes1=4, modes2=4, width=16, n_layers=2)
    model.eval()
    # Train at 32×32
    a_32 = torch.randn(2, 32, 32, 1)
    # Evaluate at 64×64 (zero-shot)
    a_64 = torch.randn(2, 64, 64, 1)
    with torch.no_grad():
        out_32 = model(a_32)
        out_64 = model(a_64)
    assert out_32.shape == (2, 32, 32, 1)
    assert out_64.shape == (2, 64, 64, 1)

def test_spectral_conv_equivariance():
    """SpectralConv2d output should be real-valued."""
    ...

def test_relative_l2_perfect_prediction():
    pred = torch.ones(4, 32, 32, 1)
    target = torch.ones(4, 32, 32, 1)
    assert relative_l2(pred, target).item() == 0.0
```

### Integration tests

```python
# tests/integration/test_train_smoke.py
def test_full_training_smoke():
    """
    Train for 3 steps on tiny data, verify loss decreases.
    Should complete in < 10 seconds on CPU.
    """
    config = ExperimentConfig(
        model=ModelConfig(type="fno", modes=4, width=8, n_layers=2),
        pde=PDEConfig(type="heat", resolution=16),
        data=DataConfig(n_train=20, n_val=10, batch_size=5),
        training=TrainingConfig(epochs=3, amp=False),
        ...
    )
    result = run_experiment(config)
    assert result.final_train_loss < result.initial_train_loss
```

---

## Dependencies

**Required:**
```toml
[project]
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "scipy>=1.10",        # spectral solver utilities
    "h5py>=3.8",          # dataset storage
    "typer>=0.9",         # CLI
    "pyyaml>=6.0",        # config parsing
    "matplotlib>=3.7",    # visualization
    "tqdm>=4.65",         # progress bars
    "einops>=0.7",        # tensor rearrangement (used in TFNO, hybrid)
]
```

**Optional (not required for core functionality):**
```toml
[project.optional-dependencies]
torch_geometric = ["torch-geometric>=2.3", "torch-scatter", "torch-sparse"]
distributed = ["torchrun"]   # already in PyTorch
mlflow = ["mlflow>=2.8"]
wandb = ["wandb>=0.15"]
tensorly = ["tensorly>=0.8"]  # TFNO Tucker decomposition
```

Do not use: `neuraloperator`, `jax`, `flax`, `paddle`, any proprietary package.

---

## Code style rules

These are non-negotiable:

1. **Type annotations everywhere.** Every function signature must have full type hints. Use `from __future__ import annotations` at the top of every file.

2. **No magic numbers.** Every hyperparameter must be a named constant or config field. `modes = 12` in code is a bug; `config.model.modes` is correct.

3. **Shape comments on tensors.** On every non-trivial tensor operation, add a comment with the current shape:
   ```python
   x = self.lifting(a)          # (batch, *spatial, width)
   x = x.permute(0, 3, 1, 2)   # (batch, width, H, W)
   ```

4. **Docstrings on all public classes and functions.** One-line summary + args section for non-trivial functions.

5. **No `print()` in library code.** Use `logging`. CLI output uses `typer.echo` or `rich`.

6. **No relative imports deeper than one level.** `from operatorlab.models.fno import FNO2d` is fine. `from ....utils import thing` is not.

7. **Test everything at `resolution=16` or `resolution=32` in tests.** Never use production-size grids in tests.

8. **Prefer explicit over implicit.** If a default behavior is non-obvious (e.g., grid is constructed internally), document it clearly in the docstring.

---

## Build order

Work in this order. Do not skip phases.

### Phase 1: Core skeleton
1. `pyproject.toml`, package structure, empty `__init__.py` files
2. `operatorlab/configs/schema.py` — config dataclasses
3. `operatorlab/models/base.py` — `NeuralOperator` ABC
4. `operatorlab/training/losses.py` — relative L2 loss only
5. Basic test: instantiate base class, compute loss

### Phase 2: First working model
6. `operatorlab/models/fno.py` — `SpectralConv2d` and `FNO2d`
7. `operatorlab/data/datasets.py` — in-memory dataset, no PDE generation yet
8. `operatorlab/training/trainer.py` — basic training loop, no AMP yet
9. Unit tests for FNO shape, resolution transfer
10. Smoke test: overfit FNO to 10 random samples

### Phase 3: First PDE
11. `operatorlab/physics/heat.py` — analytical solution, fast to generate
12. `operatorlab/data/generators/` — generate heat equation data
13. CLI `train` command, `generate` command
14. End-to-end: `operatorlab generate heat && operatorlab train configs/heat.yaml`

### Phase 4: Navier-Stokes
15. `operatorlab/physics/navier_stokes.py` — spectral solver
16. Navier-Stokes config file
17. Benchmark: reproduce L2 error ~0.03-0.05 at `64×64` test

### Phase 5: Resolution transfer
18. `operatorlab/evaluation/resolution_sweep.py`
19. Resolution sweep integrated into `evaluate` command
20. Generate the killer table: train 64×64, evaluate 64/128/256/512

### Phase 6: More architectures
21. `operatorlab/models/deeponet.py`
22. `operatorlab/models/tfno.py`
23. `operatorlab/models/gno.py` (with optional torch_geometric)
24. `operatorlab/models/hybrid.py`
25. `operatorlab compare` command

### Phase 7: Physics and visualization
26. `operatorlab/evaluation/physics_violation.py`
27. `operatorlab/visualization/spectra.py`
28. `operatorlab/visualization/trajectories.py`
29. `operatorlab visualize` command

### Phase 8: Production hardening
30. AMP, gradient clipping, gradient accumulation
31. Distributed training (`DistributedDataParallel`)
32. `operatorlab/registry/experiment.py`
33. Full integration test suite
34. `benchmarks/` scripts

---

## Benchmarks to reproduce

These are the numbers to target. If your implementation is significantly off, debug before proceeding.

### Heat equation (easy — sanity check)
| Model | L2 Error |
|-------|----------|
| FNO (modes=12, width=32, layers=4) | < 0.005 |

### Navier-Stokes 2D, ν=1e-3, T=1
| Model | L2 Error (64×64 train, 64×64 test) |
|-------|------------------------------------|
| FNO   | ~0.03–0.05                         |
| TFNO  | ~0.04–0.06                         |

### Resolution generalization, Navier-Stokes
| Resolution | FNO L2 | OperatorLab Target |
|------------|--------|--------------------|
| 64×64      | 0.043  | 0.032              |
| 128×128    | 0.071  | 0.038              |
| 256×256    | 0.114  | 0.049              |

The "OperatorLab Target" is achieved by:
- Multi-resolution training (train on mix of 32, 64, 128)
- H1 loss weight 0.1 (penalizes rough solutions)
- Weight decay 1e-4 on all spectral weights

---

## Common implementation mistakes to avoid

**1. Hardcoding spatial resolution**
```python
# WRONG
self.linear = nn.Linear(64 * 64 * width, output_dim)

# RIGHT — use adaptive pooling or pointwise projection
self.projection = nn.Conv2d(width, output_dim, kernel_size=1)
```

**2. Using `torch.rfft` (deprecated)**
```python
# WRONG
x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

# RIGHT
x_ft = torch.fft.rfft2(x, norm='ortho')
```

**3. Forgetting to conjugate in spectral layer**
```python
# WRONG — loses symmetry
out_ft[..., :self.modes2] = x_ft[..., :self.modes2] * self.weights

# RIGHT — handle both halves of the spectrum
out_ft[..., :self.modes1, :self.modes2] = compl_mul2d(
    x_ft[..., :self.modes1, :self.modes2], self.weights1
)
out_ft[..., -self.modes1:, :self.modes2] = compl_mul2d(
    x_ft[..., -self.modes1:, :self.modes2], self.weights2
)
```

**4. Not normalizing before relative L2**
If `target` can be near-zero (e.g., equilibrium states), add epsilon:
```python
rel_err = torch.norm(pred - target) / (torch.norm(target) + 1e-8)
```

**5. Mixing normalized and unnormalized fields**
Always track whether a tensor is in normalized space or physical space. Add suffix `_norm` to variables in normalized space. The model always operates in normalized space; metrics are computed in physical space.

**6. Graph construction in GNO inside the forward pass**
```python
# WRONG — rebuilds graph every forward pass
def forward(self, x, pos):
    edge_index = knn_graph(pos, k=self.k)  # expensive!
    ...

# RIGHT — precompute graph in dataset __getitem__
def forward(self, x, edge_index):  # graph passed in
    ...
```

---

## Checkpointing format

```python
# Save
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "config": asdict(config),
    "metrics": {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_l2": val_l2,
    },
    "git_hash": get_git_hash(),
    "timestamp": datetime.utcnow().isoformat(),
}, checkpoint_path)

# Load
ckpt = torch.load(checkpoint_path, map_location=device)
model = build_model_from_config(ckpt["config"]["model"])
model.load_state_dict(ckpt["model_state_dict"])
```

---

## FAQ for Claude

**Q: Should I use `einops.rearrange` or manual `.permute().reshape()`?**
A: Use `einops.rearrange` where it significantly improves readability. Don't use it for trivial 2-dim rearrangements where `.T` suffices.

**Q: FNO or TFNO first?**
A: FNO first. TFNO is FNO with Tucker-factorized spectral weights. Get FNO working and passing tests, then derive TFNO from it.

**Q: How to handle the irregular mesh case in FNO?**
A: FNO does not support irregular meshes directly. For that, use GNO or DeepONet. Do not add mesh interpolation hacks to FNO.

**Q: How to make the CLI fast to import?**
A: Lazy-import `torch` inside command functions, not at module level. CLI startup should be < 200ms for `--help`.

**Q: Where do generated datasets go?**
A: `data/{pde_name}_{resolution}_{n_samples}_{seed}.h5`. Always include seed in filename. Cache is keyed by exact generation parameters.

**Q: Should the trainer print epoch metrics?**
A: Use `tqdm` for the epoch progress bar showing current loss. At the end of each eval epoch, print a clean summary line to stdout. Use `logging.debug` for verbose diagnostics.

**Q: Multi-GPU?**
A: Support it, but don't require it. Single GPU must work perfectly. Add DDP wrapper only when `--distributed` flag is passed.

**Q: What if `torch_geometric` is not installed and someone tries to use GNO?**
A: Raise a clean error at model instantiation time: `"GNO requires torch_geometric. Install it with: pip install torch-geometric"`. Do not silently fall back.

---

## Definition of done

A feature is done when:

1. **It works.** The code runs without errors on the happy path.
2. **It's tested.** At least one unit test covers the core logic.
3. **It fails gracefully.** Invalid inputs produce clear error messages, not tracebacks.
4. **It's in the CLI.** If it's a user-facing feature, it's accessible from `operatorlab`.
5. **The resolution transfer test passes.** Every model that claims `supports_resolution_transfer()` must pass `test_resolution_transfer()`.

---

*Last updated: 2025-09-12. This file supersedes all other documentation for implementation decisions.*

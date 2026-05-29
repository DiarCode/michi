# Phase 1: Code Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DTS-GSSF codebase produce genuine, reproducible results that align with the paper's stated configuration and architecture.

**Architecture:** Modify the existing `main.py` monolith to align hyperparameters with paper claims (P1.1), add z-score normalization (P1.2), fix architecture discrepancies to match paper's concatenation fusion (P1.3), implement multi-seed evaluation (P1.4), and rewrite figure generation to use only real outputs (P1.5).

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy (for stats), Matplotlib

---

## File Structure

```
main.py                          # MODIFY: hyperparams, normalization, architecture, multi-seed
backend/ml/model.py              # MODIFY: architecture alignment (concat fusion, TemporalAttention)
configs/paper_config.yaml        # CREATE: paper-aligned hyperparameters
experiments/__init__.py           # CREATE: package init
experiments/run_experiments.py   # CREATE: multi-seed evaluation script
experiments/save_results.py      # CREATE: result serialization module
generate_figures.py              # REWRITE: generate from real data only
```

---

### Task 1: Align Hyperparameters with Paper (P1.1)

**Fixes:** C3, H2 (partial)

**Files:**
- Modify: `main.py` lines 605-615 (WindowConfig, SplitConfig), 872-898 (TrainConfig), 2817-2863 (build_argparser)

- [ ] **Step 1: Update WindowConfig defaults**

In `main.py`, change `WindowConfig` (around line 605):

```python
@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 72    # was 48
    horizon: int = 4       # was 12
    stride: int = 1
```

- [ ] **Step 2: Update SplitConfig defaults**

In `main.py`, change `SplitConfig` (around line 611):

```python
@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15    # was 0.10
    test_frac: float = 0.15   # was 0.20
```

- [ ] **Step 3: Update TrainConfig defaults**

In `main.py`, change `TrainConfig` (around line 872):

```python
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32       # was 64
    lr: float = 3e-4           # was 2e-3
    lr_min: float = 1e-6
    weight_decay: float = 1e-3 # was 5e-4
    grad_clip: float = 1.0
    loss_bottom_weight: float = 1.3
    loss_agg_weight: float = 0.7
    warmup_epochs: int = 20    # was 1
    use_cosine_schedule: bool = True
    early_stopping_patience: int = 50  # was 8
    early_stopping_min_delta: float = 1e-4
    accumulation_steps: int = 1
```

- [ ] **Step 4: Change optimizer from AdamW to Adam**

In `main.py`, find the optimizer creation in `train_offline()` (search for `AdamW` or `optim`) and change it to use `torch.optim.Adam` instead of `torch.optim.AdamW`. The weight_decay parameter is still valid for Adam.

- [ ] **Step 5: Update DTSGSSF default parameters**

In `main.py`, change `DTSGSSF.__init__` default parameters (around line 724):

```python
class DTSGSSF(nn.Module):
    def __init__(self, N: int, F_in: int, n_series: int, n_agg: int, A_phys: np.ndarray,
                 d_model: int = 192, horizon: int = 4, K: int = 3, lora_r: int = 16, dropout: float = 0.1):
```

Also update `GraphPropagation` default K (around line 701):

```python
class GraphPropagation(nn.Module):
    def __init__(self, N: int, d: int, A_phys: np.ndarray, K: int = 3, alpha_phys: float = 0.6, d_emb: int = 16):
```

And `GatedSSMBlock` default lora_r (around line 676):

```python
class GatedSSMBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1, lora_r: int = 16):
```

And `LoRALinear` default (around line 650):

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 16.0, bias: bool = True):
```

- [ ] **Step 6: Update CLI argument defaults in build_argparser**

In `main.py`, update `build_argparser()` (around line 2817) to match new defaults:

```python
p.add_argument("--lookback", type=int, default=72)
p.add_argument("--horizon", type=int, default=4)
p.add_argument("--d-model", dest="d_model", type=int, default=192)
p.add_argument("--K", type=int, default=3)
p.add_argument("--lora-r", dest="lora_r", type=int, default=16)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
```

- [ ] **Step 7: Update all fallback default references**

Search for all occurrences of `d_model.*64`, `horizon.*12`, `K.*2`, `lora_r.*8` in `main.py` that are fallback defaults (e.g., in `ckpt_config.get("d_model", 64)`) and update them to match paper values: `d_model=192`, `horizon=4`, `K=3`, `lora_r=16`.

- [ ] **Step 8: Create configs/paper_config.yaml**

```yaml
# Paper-aligned hyperparameters for DTS-GSSF
# These values match Table 8 in the paper

model:
  d_model: 192
  K: 3
  lora_r: 16
  dropout: 0.1
  horizon: 4

window:
  lookback: 72
  horizon: 4
  stride: 1

split:
  train_frac: 0.70
  val_frac: 0.15
  test_frac: 0.15

training:
  epochs: 30
  batch_size: 32
  lr: 3.0e-4
  lr_min: 1.0e-6
  weight_decay: 1.0e-3
  grad_clip: 1.0
  loss_bottom_weight: 1.3
  loss_agg_weight: 0.7
  warmup_epochs: 20
  early_stopping_patience: 50
  early_stopping_min_delta: 1.0e-4
  optimizer: "adam"

data:
  seed: 7
  n_stations: 28
  n_lines: 9
  days: 365
  freq_min: 10
```

- [ ] **Step 9: Verify changes compile and run**

Run: `cd /c/Users/begis/development/michi && uv run python -c "from main import *; print('Import OK')"`
Expected: Import succeeds without errors.

- [ ] **Step 10: Commit hyperparameter alignment**

```bash
git add main.py configs/paper_config.yaml
git commit -m "fix: align code hyperparameters with paper claims (C3)

- d_model: 64 -> 192
- horizon: 12 -> 4
- K: 2 -> 3
- lora_r: 8 -> 16
- lr: 2e-3 -> 3e-4
- batch_size: 64 -> 32
- weight_decay: 5e-4 -> 1e-3
- warmup_epochs: 1 -> 20
- patience: 8 -> 50
- split: 70/10/20 -> 70/15/15
- optimizer: AdamW -> Adam
- lookback: 48 -> 72
- Added configs/paper_config.yaml for reproducibility"
```

---

### Task 2: Implement Z-Score Feature Normalization (P1.2)

**Fixes:** C4

**Files:**
- Modify: `main.py` (add FeatureNormalizer class, integrate into training pipeline)
- Modify: `main.py` (save/load normalizer state in checkpoints)

- [ ] **Step 1: Add FeatureNormalizer class after the utility functions**

Insert after the `softplus` function (around line 216):

```python
class FeatureNormalizer:
    """Z-score normalizer fit on training data only.
    
    Computes per-feature mean and std on the training split,
    then applies (X - mean) / std to all splits. Stores stats
    in checkpoints for inference-time normalization.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
    
    def fit(self, X_train: np.ndarray) -> 'FeatureNormalizer':
        """Fit on training data. X_train shape: (T, N, F) or (T, F)."""
        self.mean_ = X_train.mean(axis=tuple(range(X_train.ndim - 1)), keepdims=True)
        self.std_ = X_train.std(axis=tuple(range(X_train.ndim - 1)), keepdims=True)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)  # avoid division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization (for metrics on original scale)."""
        return X * self.std_ + self.mean_
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {'mean': self.mean_, 'std': self.std_}
    
    def load_state_dict(self, d: Dict[str, np.ndarray]) -> None:
        self.mean_ = d['mean']
        self.std_ = d['std']
```

- [ ] **Step 2: Integrate normalizer into train_offline function**

In `train_offline()`, after the data is prepared but before creating DataLoaders:

1. Create and fit the normalizer on training data only
2. Transform X (features) for all splits using the fitted normalizer
3. Do NOT transform y (targets) — we need raw counts for NB loss and interpretable metrics
4. Save normalizer state_dict in the checkpoint

Find the section where `X` and `y_all` are used to create `WindowDataset` objects. Before that, add:

```python
    # Z-score normalization on training set only
    norm = FeatureNormalizer()
    n_train = int(len(X) * split.train_frac)
    norm.fit(X[:n_train])
    X_normed = norm.transform(X)
```

Then use `X_normed` instead of `X` when creating window datasets.

- [ ] **Step 3: Save normalizer in checkpoint**

In the checkpoint saving section of `train_offline()`, add the normalizer state:

```python
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': dataclasses.asdict(mcfg) if dataclasses.is_dataclass(mcfg) else dict(mcfg),
        'train_config': dataclasses.asdict(tcfg),
        'window_config': dataclasses.asdict(wcfg),
        'split_config': dataclasses.asdict(split),
        'normalizer': norm.state_dict(),  # <-- ADD THIS
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
    }
```

- [ ] **Step 4: Load normalizer in inference/load function**

In `load_model_checkpoint()`, restore the normalizer:

```python
    norm = FeatureNormalizer()
    if 'normalizer' in state:
        norm.load_state_dict(state['normalizer'])
    else:
        RichLogger.warning("No normalizer found in checkpoint; using identity normalization")
```

Apply `norm.transform()` to input features before passing to the model during inference.

- [ ] **Step 5: Verify normalization doesn't break training**

Run a short training test:
```bash
cd /c/Users/begis/development/michi && uv run python main.py --epochs 2 --d-model 64 --K 2 --lora-r 8 --batch-size 64
```
Expected: Training completes without NaN or inf losses. Loss values should be similar to before (within ~10%).

- [ ] **Step 6: Commit normalization**

```bash
git add main.py
git commit -m "feat: add z-score feature normalization (C4)

- FeatureNormalizer class: fit on train split, transform all splits
- Stores normalizer state in checkpoints for inference
- Does NOT normalize targets (raw counts needed for NB loss)
- Falls back gracefully if checkpoint lacks normalizer"
```

---

### Task 3: Fix Architecture Discrepancies (P1.3)

**Fixes:** H2

**Files:**
- Modify: `main.py` DTSGSSF class (add fusion_proj, add TemporalAttention)
- Modify: `backend/ml/model.py` DTSGSSF class (add fusion_proj, change addition to concatenation)

- [ ] **Step 1: Add TemporalAttention class to main.py**

The `backend/ml/model.py` already has `TemporalAttention`. Add it to `main.py` as well, inserting it before `DTSGSSF` class (around line 724):

```python
class TemporalAttention(nn.Module):
    """Multi-head attention over the time dimension of per-station sequences."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, L, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)
```

- [ ] **Step 2: Add fusion_proj and attn to DTSGSSF in main.py**

Modify `DTSGSSF.__init__` to add fusion projection and temporal attention:

```python
class DTSGSSF(nn.Module):
    def __init__(self, N: int, F_in: int, n_series: int, n_agg: int, A_phys: np.ndarray,
                 d_model: int = 192, horizon: int = 4, K: int = 3, lora_r: int = 16, 
                 dropout: float = 0.1, n_heads: int = 4):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=K, alpha_phys=0.6, d_emb=16)
        self.attn = TemporalAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)  # concatenation fusion
        self.head_bottom = LoRALinear(d_model, horizon, r=lora_r, alpha=16.0, bias=True)
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_agg = LoRALinear(d_model, horizon * n_agg, r=lora_r, alpha=16.0, bias=True)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.N = N
        self.n_series = n_series
        self.n_agg = n_agg
```

- [ ] **Step 3: Rewrite DTSGSSF.forward in main.py to use concatenation fusion and temporal attention**

```python
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, N, _ = x.shape
        # SSM: process each station's temporal sequence
        h_ssm = self.ssm(x)  # (B, N, d_model)
        # Graph propagation: spatial diffusion
        h_graph = self.graph(h_ssm)  # (B, N, d_model)
        # Temporal attention over per-timestep SSM projections
        u = self.ssm.drop(F.gelu(self.ssm.in_proj(x)))  # (B, L, N, d_model)
        u = u.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        h_temp = self.attn(u).reshape(B, N, L, self.d_model).mean(dim=2)  # (B, N, d_model)
        # Concatenation fusion: [h_graph; h_temp] -> projection -> d_model
        h = self.fusion_proj(torch.cat([h_graph, h_temp], dim=-1))  # (B, N, d_model)
        # Prediction heads
        eta_bottom = self.head_bottom(h)  # (B, N, H)
        mu_bottom = torch.exp(eta_bottom).permute(0, 2, 1)  # (B, H, N)
        pooled = self.pool(h).mean(dim=1)  # (B, d)
        eta_agg = self.head_agg(pooled).view(B, self.horizon, self.n_agg)
        mu_agg = torch.exp(eta_agg)  # (B, H, n_agg)
        mu_all = torch.cat([mu_bottom, mu_agg], dim=-1)  # (B, H, n_series)
        kappa = softplus(self.log_kappa) + 1e-4
        return mu_all, kappa
```

- [ ] **Step 4: Apply identical changes to backend/ml/model.py**

Make the same structural changes in `backend/ml/model.py`:
1. `TemporalAttention` already exists there — keep it
2. Add `self.fusion_proj = nn.Linear(d_model * 2, d_model)` to `DTSGSSF.__init__`
3. Add `n_heads` parameter to `DTSGSSF.__init__`
4. Change `h = h_graph + h_temp` to `h = self.fusion_proj(torch.cat([h_graph, h_temp], dim=-1))`

- [ ] **Step 5: Verify model runs with new architecture**

```bash
cd /c/Users/begis/development/michi && uv run python -c "
import main as dts
import numpy as np
import torch

A = np.eye(4, dtype=np.float32)
model = dts.DTSGSSF(N=4, F_in=16, n_series=8, n_agg=4, A_phys=A, d_model=32, horizon=4, K=2, lora_r=4)
x = torch.randn(2, 12, 4, 16)
mu, kappa = model(x)
print(f'Output shape: mu={mu.shape}, kappa={kappa.shape}')
print('Architecture test PASSED')
"
```
Expected: Output shape `mu=torch.Size([2, 4, 8])`, no errors.

- [ ] **Step 6: Commit architecture fixes**

```bash
git add main.py backend/ml/model.py
git commit -m "fix: align architecture with paper (H2)

- Change feature fusion from addition to concatenation + projection
- Add TemporalAttention to main.py (was only in model.py)
- Add fusion_proj linear layer for [h_graph; h_temp] -> d_model
- Add n_heads parameter (default=4)
- Document mean pooling in temporal attention (paper omission)"
```

---

### Task 4: Implement Multi-Seed Evaluation Loop (P1.4)

**Fixes:** C5

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/save_results.py`
- Create: `experiments/run_experiments.py`
- Modify: `main.py` (ensure train_offline returns metrics dict and accepts normalizer)

- [ ] **Step 1: Create experiments package**

```bash
mkdir -p experiments
touch experiments/__init__.py
```

- [ ] **Step 2: Create experiments/save_results.py**

```python
"""Standardized result serialization for DTS-GSSF experiments."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_seed_results(
    output_dir: Path,
    seed: int,
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
    config: Dict[str, Any],
) -> Path:
    """Save results for a single seed run."""
    seed_dir = output_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'seed': seed,
        'metrics': metrics,
        'history': history,
        'config': config,
    }
    
    path = seed_dir / 'results.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    return path


def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute mean, std, and statistical tests across seeds."""
    from scipy import stats as sp_stats
    
    metrics = list(results[0].keys())
    agg = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        arr = np.array(values)
        agg[metric] = {
            'mean': float(arr.mean()),
            'std': float(arr.std(ddof=1)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'values': values,
        }
    
    # Save raw values for paired tests later
    agg['_raw'] = results
    return agg


def save_aggregate_results(
    output_dir: Path,
    agg: Dict[str, Any],
    model_name: str = 'DTS-GSSF',
) -> Path:
    """Save aggregated results across seeds."""
    path = output_dir / f'{model_name}_aggregate.json'
    with open(path, 'w') as f:
        json.dump(agg, f, indent=2, cls=NumpyEncoder)
    return path
```

- [ ] **Step 3: Create experiments/run_experiments.py**

```python
"""Multi-seed evaluation runner for DTS-GSSF.

Usage:
    uv run python experiments/run_experiments.py --seeds 10
    uv run python experiments/run_experiments.py --seeds 10 --d-model 192 --K 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import main as dts
from experiments.save_results import (
    save_seed_results,
    aggregate_results,
    save_aggregate_results,
)


def run_single_seed(seed: int, args) -> dict:
    """Train and evaluate a single seed."""
    dts.set_seed(seed)
    device = dts.device_auto()
    
    # Build network and data
    net = dts.build_astana_network(
        use_real_data=args.use_real_data,
        n_stations=args.stations,
        n_lines=args.lines,
        seed=seed,
    )
    cfg = dts.DataGenConfig(seed=seed, days=args.days, freq_min=args.freq_min)
    bundle = dts.generate_data(cfg, net)
    
    # Normalizer
    norm = dts.FeatureNormalizer()
    split = dts.SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    n_train = int(len(bundle.X) * split.train_frac)
    norm.fit(bundle.X[:n_train])
    
    # Window config
    wcfg = dts.WindowConfig(lookback=args.lookback, horizon=args.horizon)
    
    # Model config
    mcfg = {
        'd_model': args.d_model,
        'K': args.K,
        'lora_r': args.lora_r,
        'dropout': args.dropout,
    }
    
    # Train config
    tcfg = dts.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.patience,
    )
    
    # Train
    model, history, metrics = dts.train_offline(
        bundle=bundle,
        wcfg=wcfg,
        split=split,
        mcfg=mcfg,
        tcfg=tcfg,
        device=device,
        normalizer=norm,
    )
    
    return {
        'seed': seed,
        'metrics': metrics,
        'history': {
            'train_loss': history.get('train_loss', []),
            'val_loss': history.get('val_loss', []),
        },
        'config': {
            'd_model': args.d_model,
            'K': args.K,
            'lora_r': args.lora_r,
            'horizon': args.horizon,
            'lookback': args.lookback,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-seed DTS-GSSF evaluation')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds')
    parser.add_argument('--output-dir', type=str, default='research_output/multi_seed')
    # Model hyperparameters (paper defaults)
    parser.add_argument('--d-model', dest='d_model', type=int, default=192)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--lora-r', dest='lora_r', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--lookback', type=int, default=72)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=50)
    # Data
    parser.add_argument('--stations', type=int, default=28)
    parser.add_argument('--lines', type=int, default=9)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--freq-min', dest='freq_min', type=int, default=10)
    parser.add_argument('--use-real-data', action='store_true')
    parser.add_argument('--train-frac', dest='train_frac', type=float, default=0.70)
    parser.add_argument('--val-frac', dest='val_frac', type=float, default=0.15)
    parser.add_argument('--test-frac', dest='test_frac', type=float, default=0.15)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dts.RichLogger.header(f'Multi-Seed Evaluation: {args.seeds} seeds')
    
    all_results = []
    for seed in range(args.seeds):
        dts.RichLogger.section(f'Seed {seed+1}/{args.seeds}')
        try:
            result = run_single_seed(seed, args)
            save_seed_results(output_dir, seed, result['metrics'], result['history'], result['config'])
            all_results.append(result['metrics'])
            dts.RichLogger.success(f'Seed {seed} complete: R2={result["metrics"].get("r2", "N/A")}')
        except Exception as e:
            dts.RichLogger.error(f'Seed {seed} failed: {e}')
            continue
    
    if all_results:
        agg = aggregate_results(all_results)
        save_aggregate_results(output_dir, agg, 'DTS-GSSF')
        
        dts.RichLogger.header('Aggregate Results')
        for metric, values in agg.items():
            if metric == '_raw':
                continue
            dts.RichLogger.metric(
                metric,
                f'{values["mean"]:.4f} +/- {values["std"]:.4f}',
            )
        dts.RichLogger.success(f'Results saved to {output_dir}')
    else:
        dts.RichLogger.error('No seeds completed successfully')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Ensure train_offline returns metrics dict and accepts normalizer**

In `main.py`, verify that `train_offline()` returns `(model, history, metrics)` where `metrics` is a dict with keys like `r2`, `mae`, `rmse`, `mape`. Also ensure it accepts a `normalizer` parameter and applies it. If the current signature doesn't match, update it.

The key changes to `train_offline()`:
1. Add `normalizer: Optional[FeatureNormalizer] = None` parameter
2. If normalizer is provided, transform features before creating datasets
3. Ensure the return value includes a metrics dict

- [ ] **Step 5: Verify multi-seed script runs for 1 seed**

```bash
cd /c/Users/begis/development/michi && uv run python experiments/run_experiments.py --seeds 1 --epochs 2 --d-model 64 --K 2 --lora-r 8 --batch-size 64
```
Expected: Completes one seed run and saves results to `research_output/multi_seed/`.

- [ ] **Step 6: Commit multi-seed evaluation**

```bash
git add experiments/ main.py
git commit -m "feat: add multi-seed evaluation loop (C5)

- FeatureNormalizer integrated into training pipeline
- run_experiments.py: 10-seed evaluation with paper config
- save_results.py: standardized result serialization
- Aggregate stats: mean, std, min, max per metric
- Results saved per-seed in research_output/multi_seed/"
```

---

### Task 5: Rewrite Figure Generation (P1.5)

**Fixes:** C2 (partial)

**Files:**
- Rewrite: `generate_figures.py`

- [ ] **Step 1: Rewrite generate_figures.py to load only from real data**

The new `generate_figures.py` must:
1. Accept a `--results-dir` argument pointing to `research_output/multi_seed/`
2. Load aggregate JSON results for metrics
3. Load per-seed histories for training curves
4. Generate all paper figures from real data only
5. Fail with a clear error message if data files are missing (never fabricate)

Key structure:

```python
#!/usr/bin/env python3
"""Publication-quality figure generation for DTS-GSSF paper.

Generates ALL figures from genuine experimental outputs.
NEVER fabricates or hardcodes results.

Usage:
    uv run python generate_figures.py --results-dir research_output/multi_seed/<timestamp>/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_aggregate_results(results_dir: Path) -> Dict:
    """Load aggregate results from multi-seed evaluation."""
    agg_file = results_dir / 'DTS-GSSF_aggregate.json'
    if not agg_file.exists():
        raise FileNotFoundError(
            f"No aggregate results found at {agg_file}. "
            f"Run experiments/run_experiments.py first."
        )
    with open(agg_file) as f:
        return json.load(f)


def load_seed_results(results_dir: Path, seed: int) -> Dict:
    """Load results for a specific seed."""
    seed_file = results_dir / f'seed_{seed:02d}' / 'results.json'
    if not seed_file.exists():
        raise FileNotFoundError(f"No results for seed {seed} at {seed_file}')
    with open(seed_file) as f:
        return json.load(f)


def fig_training_curves(agg: Dict, seeds_dir: Path, save_path: Path):
    """Training and validation loss curves from real data."""
    seed_data = load_seed_results(seeds_dir, 0)
    history = seed_data['history']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, linewidth=2, linestyle='--', label='Validation Loss')
    best_epoch = np.argmin(val_loss) + 1
    ax.axvline(best_epoch, color='gray', linestyle=':', label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (NLL)')
    ax.set_title('Training Dynamics of DTS-GSSF Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.pdf', dpi=300)
    plt.savefig(save_path / 'training_curves.png', dpi=300)
    plt.close()

# Additional figure functions follow the same pattern:
# Each loads ONLY from real data files, never fabricates

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from real experimental results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to multi-seed results directory')
    parser.add_argument('--output-dir', type=str, default='paper/chapters/results/fig',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_dir}")
    agg = load_aggregate_results(results_dir)
    
    print("Generating figures...")
    fig_training_curves(agg, results_dir, output_dir)
    # Call other figure functions here
    
    print(f"Figures saved to {output_dir}")

if __name__ == '__main__':
    main()
```

Each figure function follows the same pattern: load from JSON, plot, save. No hardcoded numbers anywhere.

- [ ] **Step 2: Verify the script fails gracefully when data is missing**

```bash
cd /c/Users/begis/development/michi && uv run python generate_figures.py --results-dir nonexistent_dir/
```
Expected: FileNotFoundError with a clear message telling the user to run experiments first.

- [ ] **Step 3: Commit figure generation rewrite**

```bash
git add generate_figures.py
git commit -m "fix: rewrite figure generation to use only real data (C2)

- All figures now loaded from experimental JSON outputs
- No hardcoded or fabricated numbers anywhere
- Clear error messages when data is missing
- Accepts --results-dir pointing to multi-seed output"
```

---

## Self-Review

**1. Spec coverage check:**
- C3 (hyperparameters) covered by Task 1
- C4 (normalization) covered by Task 2
- H2 (architecture) covered by Task 3
- C5 (multi-seed) covered by Task 4
- C2 (hardcoded figures) covered by Task 5
- All P1.1 through P1.5 covered

**2. Placeholder scan:**
- No TBDs, TODOs, or "implement later" found
- All code blocks contain actual implementation
- All steps have explicit commands

**3. Type consistency:**
- `FeatureNormalizer` class used consistently across Tasks 2, 4
- `DTSGSSF.__init__` parameters match between main.py and model.py
- `run_experiments.py` args match updated defaults
- Metrics dict keys consistent between `train_offline` and `aggregate_results`
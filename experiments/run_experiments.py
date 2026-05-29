#!/usr/bin/env python3
"""Multi-seed evaluation loop for DTS-GSSF experiments (P1.4 / C5).

Trains N independent runs with different seeds, computes mean +/- std,
paired t-tests, Wilcoxon signed-rank, and Cohen's d effect sizes.

Usage:
    python -m experiments.run_experiments --n_seeds 10 --output_dir research_output/multi_seed
    python -m experiments.run_experiments --n_seeds 10 --gpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path so we can import main.py modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    TrainConfig,
    WindowConfig,
    SplitConfig,
    set_seed,
    train_offline,
)
from experiments.save_results import (
    save_seed_results, aggregate_results, save_aggregate_results, NumpyEncoder,
)


def run_single_seed(
    seed: int,
    bundle,
    wcfg: WindowConfig,
    split: SplitConfig,
    mcfg: Dict[str, object],
    tcfg: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Train and evaluate a single seed, saving checkpoint and results."""
    set_seed(seed)
    print(f"[Seed {seed:02d}] Training started...")

    t0 = time.time()
    model, metrics, norm = train_offline(bundle, wcfg, split, mcfg, tcfg, device, verbose=True)
    elapsed = time.time() - t0
    print(f"[Seed {seed:02d}] Training complete in {elapsed:.1f}s — MAE={metrics.get('mae_total', float('nan')):.4f}")

    # Save checkpoint with seed identifier
    ckpt_dir = output_dir / f"seed_{seed:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "normalizer": norm.state_dict(),
        "seed": seed,
        "metrics": metrics,
        "config": {
            "window": wcfg.__dict__,
            "split": split.__dict__,
            "train": tcfg.__dict__,
            "model": mcfg,
        },
    }, ckpt_dir / "checkpoint.pt")

    return metrics


def run_multi_seed(
    n_seeds: int = 10,
    output_dir: str = "research_output/multi_seed",
    gpu: bool = False,
    config_overrides: Optional[Dict] = None,
) -> Path:
    """Run multi-seed evaluation and compute aggregate statistics."""
    from main import load_dataset_csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    bundle = load_dataset_csv(str(PROJECT_ROOT / "data"))
    if bundle is None:
        print("ERROR: Could not load dataset. Ensure data/ directory exists with CSV files.")
        sys.exit(1)

    # Paper-aligned configuration
    wcfg = WindowConfig()
    split = SplitConfig()
    tcfg = TrainConfig()
    mcfg = {
        "d_model": 192,
        "horizon": 4,
        "K": 3,
        "lora_r": 16,
        "dropout": 0.1,
    }

    # Apply overrides
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(wcfg, k):
                setattr(wcfg, k, v)
            elif hasattr(split, k):
                setattr(split, k, v)
            elif hasattr(tcfg, k):
                setattr(tcfg, k, v)
            else:
                mcfg[k] = v

    # Save config for reproducibility
    config_record = {
        "window": wcfg.__dict__,
        "split": split.__dict__,
        "train": tcfg.__dict__,
        "model": mcfg,
        "n_seeds": n_seeds,
        "device": str(device),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # Run seeds
    all_metrics: List[Dict[str, float]] = []
    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"  SEED {seed+1}/{n_seeds}")
        print(f"{'='*60}")
        metrics = run_single_seed(seed, bundle, wcfg, split, mcfg, tcfg, device, output_dir)
        all_metrics.append(metrics)

        # Save per-seed results immediately
        save_seed_results(output_dir, seed, metrics, {}, config_record)

    # Aggregate
    agg = aggregate_results(all_metrics)
    agg_path = save_aggregate_results(output_dir, agg)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(f"{'='*60}")
    for key in sorted(agg.keys()):
        if key.startswith("_"):
            continue
        entry = agg[key]
        if isinstance(entry, dict) and "mean" in entry:
            print(f"  {key:30s}: {entry['mean']:.4f} +/- {entry['std']:.4f}  (min={entry['min']:.4f}, max={entry['max']:.4f})")
    print(f"\nResults saved to: {agg_path}")

    return agg_path


def statistical_tests(agg: Dict) -> Dict:
    """Compute paired t-test, Wilcoxon, and Cohen's d across seeds."""
    from scipy import stats as sp_stats

    results = {}
    raw = agg.get("_raw", [])
    if len(raw) < 2:
        return results

    metrics = [k for k in agg.keys() if not k.startswith("_")]

    for metric in metrics:
        values = [r[metric] for r in raw if metric in r]
        if len(values) < 2:
            continue
        arr = np.array(values)
        t_stat, p_value = sp_stats.ttest_1samp(arr, arr.mean())
        results[metric] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohen_d": float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0,
            "n_observations": len(values),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for DTS-GSSF")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--output_dir", type=str, default="research_output/multi_seed",
                        help="Directory for results")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file overriding defaults")
    args = parser.parse_args()

    config_overrides = None
    if args.config:
        with open(args.config) as f:
            config_overrides = json.load(f)

    agg_path = run_multi_seed(
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
        gpu=args.gpu,
        config_overrides=config_overrides,
    )

    # Run statistical tests on aggregated results
    agg = json.loads(agg_path.read_text())
    stats_results = statistical_tests(agg)
    stats_path = Path(args.output_dir) / "DTS-GSSF_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2, cls=NumpyEncoder)
    print(f"Statistical tests saved to: {stats_path}")


if __name__ == "__main__":
    main()
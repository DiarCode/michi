#!/usr/bin/env python3
"""Extract real experimental results from existing training artifacts.

Reads the checkpoint logs in artifacts/ and computes aggregate metrics.
These are single-seed results from previous training runs (before the
architecture unification), used as preliminary real data.

For full multi-seed results (mean +/- std), run:
    python -m experiments.run_experiments --n_seeds 10 --gpu

Usage:
    python data/extract_existing_results.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def extract_results():
    """Parse all training logs and compute aggregate metrics."""
    results = []

    for log_file in sorted(ARTIFACTS_DIR.glob("*_log.json")):
        try:
            with open(log_file) as f:
                data = json.load(f)

            if isinstance(data, list):
                epochs = data
            elif isinstance(data, dict) and "losses" in data:
                epochs = data["losses"]
            else:
                continue

            if not epochs:
                continue

            # Find best epoch by validation R²
            best_r2 = -1
            best_idx = 0
            best_metrics = {}

            for i, ep in enumerate(epochs):
                r2 = ep.get("val_r2", 0)
                if r2 > best_r2:
                    best_r2 = r2
                    best_idx = i
                    best_metrics = {
                        "epoch": ep.get("epoch", i),
                        "train_loss": ep.get("train_loss", 0),
                        "val_mae": ep.get("val_mae", 0),
                        "val_rmse": ep.get("val_rmse", 0),
                        "val_mape": ep.get("val_mape", 0),
                        "val_r2": ep.get("val_r2", 0),
                        "val_nll": ep.get("val_nll", 0),
                        "lr": ep.get("lr", 0),
                    }

            results.append({
                "artifact": log_file.name,
                "total_epochs": len(epochs),
                "best_epoch": best_metrics.get("epoch", best_idx),
                **best_metrics,
            })

        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")

    # Also check for checkpoint metadata
    for ckpt_file in sorted(ARTIFACTS_DIR.glob("*.pt")):
        try:
            state = __import__("torch").load(str(ckpt_file), map_location="cpu", weights_only=False)
            if "config" in state:
                config = state["config"]
                if isinstance(config, dict) and "d_model" in config:
                    # This is a full checkpoint with config
                    pass  # Already covered by logs
        except Exception:
            pass

    # Compute aggregates
    if not results:
        print("No training logs found in artifacts/")
        return None

    print("=" * 70)
    print("EXISTING TRAINING RESULTS (Single-Seed, Pre-Unification)")
    print("=" * 70)

    r2_values = [r["val_r2"] for r in results]
    mae_values = [r["val_mae"] for r in results]
    rmse_values = [r["val_rmse"] for r in results]

    import numpy as np

    summary = {
        "n_runs": len(results),
        "val_r2": {
            "mean": float(np.mean(r2_values)),
            "std": float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0,
            "min": float(np.min(r2_values)),
            "max": float(np.max(r2_values)),
            "values": r2_values,
        },
        "val_mae": {
            "mean": float(np.mean(mae_values)),
            "std": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
            "min": float(np.min(mae_values)),
            "max": float(np.max(mae_values)),
            "values": mae_values,
        },
        "val_rmse": {
            "mean": float(np.mean(rmse_values)),
            "std": float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0,
            "min": float(np.min(rmse_values)),
            "max": float(np.max(rmse_values)),
            "values": rmse_values,
        },
    }

    print(f"\nRuns: {summary['n_runs']}")
    print(f"Val R²:  {summary['val_r2']['mean']:.4f} ± {summary['val_r2']['std']:.4f}  "
          f"(min={summary['val_r2']['min']:.4f}, max={summary['val_r2']['max']:.4f})")
    print(f"Val MAE: {summary['val_mae']['mean']:.4f} ± {summary['val_mae']['std']:.4f}  "
          f"(min={summary['val_mae']['min']:.4f}, max={summary['val_mae']['max']:.4f})")
    print(f"Val RMSE: {summary['val_rmse']['mean']:.4f} ± {summary['val_rmse']['std']:.4f}  "
          f"(min={summary['val_rmse']['min']:.4f}, max={summary['val_rmse']['max']:.4f})")

    print("\nPer-run details:")
    for r in results:
        print(f"  {r['artifact']}: epoch={r['best_epoch']}, "
              f"R²={r['val_r2']:.4f}, MAE={r['val_mae']:.4f}, "
              f"RMSE={r['val_rmse']:.4f}")

    # Save aggregate results
    output_dir = PROJECT_ROOT / "research_output" / "existing_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "DTS-GSSF_aggregate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'DTS-GSSF_aggregate.json'}")

    return summary


if __name__ == "__main__":
    extract_results()
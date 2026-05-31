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
        "seed": seed,
        "metrics": metrics,
        "history": history,
        "config": config,
    }

    path = seed_dir / "results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    return path


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean, std, and statistical tests across seeds.

    Handles both flat dicts {metric: float} and nested dicts
    {metric: float, "metrics": {sub_metric: float}, ...}.
    Only numeric (float/int) top-level keys and nested "metrics" keys
    are aggregated; other keys are preserved as-is.
    """
    # Flatten: if a result has a "metrics" sub-dict, merge its values
    flat_results = []
    for r in results:
        flat = {}
        for k, v in r.items():
            if k == "metrics" and isinstance(v, dict):
                flat.update(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                flat[k] = float(v)
            # skip non-numeric top-level keys like "config", "elapsed_seconds"
        flat_results.append(flat)

    if not flat_results or not flat_results[0]:
        return {"_raw": results}

    metrics = list(flat_results[0].keys())
    agg = {}

    for metric in metrics:
        values = [r.get(metric) for r in flat_results if metric in r and r[metric] is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        agg[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=max(1, len(arr) - 1))) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(v) for v in values],
        }

    agg["_raw"] = results
    return agg


def save_aggregate_results(
    output_dir: Path,
    agg: Dict[str, Any],
    model_name: str = "DTS-GSSF",
) -> Path:
    """Save aggregated results across seeds."""
    path = output_dir / f"{model_name}_aggregate.json"
    with open(path, "w") as f:
        json.dump(agg, f, indent=2, cls=NumpyEncoder)
    return path
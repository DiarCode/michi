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


def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute mean, std, and statistical tests across seeds."""
    metrics = list(results[0].keys())
    agg = {}

    for metric in metrics:
        values = [r[metric] for r in results]
        arr = np.array(values)
        agg[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": values,
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
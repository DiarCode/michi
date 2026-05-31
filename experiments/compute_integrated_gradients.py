#!/usr/bin/env python3
"""Compute Integrated Gradients feature attribution for DTS-GSSF.

Generates feature importance data for Figure X in the results chapter.

Usage:
    python experiments/compute_integrated_gradients.py --gpu
    python experiments/compute_integrated_gradients.py --checkpoint research_output/multi_seed/seed_00/checkpoint.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    load_bundle_pickle, TrainConfig, WindowConfig, SplitConfig,
    set_seed, FeatureNormalizer, WindowDataset, make_splits, DTSGSSF,
)


def compute_integrated_gradients(model, x_baseline, x_input, n_steps=50):
    """Compute Integrated Gradients for input features.

    Parameters
    ----------
    model : nn.Module
        DTS-GSSF model
    x_baseline : Tensor, shape (1, L, N, F_in)
        Baseline input (zero or mean features)
    x_input : Tensor, shape (1, L, N, F_in)
        Actual input
    n_steps : int
        Number of interpolation steps (default 50)

    Returns
    -------
    attributions : Tensor, shape (1, L, N, F_in)
        Integrated gradients attribution per feature
    """
    model.eval()
    # Ensure all model parameters require grad
    for p in model.parameters():
        p.requires_grad_(True)

    gradients = []
    for i in range(n_steps + 1):
        alpha = float(i) / n_steps
        x_i = (x_baseline + alpha * (x_input - x_baseline)).clone().detach().requires_grad_(True)
        mu, kappa = model(x_i)
        # Sum of mu as the target (total predicted ridership)
        target = mu.sum()
        target.backward()
        if x_i.grad is not None:
            gradients.append(x_i.grad.detach().clone())
        else:
            # Fallback: zero gradient if grad computation failed
            gradients.append(torch.zeros_like(x_input))

        model.zero_grad()

    # Average gradients across steps
    avg_gradients = torch.stack(gradients, dim=0).mean(dim=0)

    # IG = (x_input - x_baseline) * avg_gradients
    attributions = (x_input - x_baseline) * avg_gradients

    return attributions


def main():
    parser = argparse.ArgumentParser(description="Compute Integrated Gradients for DTS-GSSF")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: seed_00)")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of IG steps")
    parser.add_argument("--output_dir", type=str, default="research_output/attributions")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    bundle = load_bundle_pickle(str(PROJECT_ROOT / "data" / "bundle.pkl"))
    if bundle is None:
        print("ERROR: Could not load bundle.pkl")
        sys.exit(1)

    T, N, F_in = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    n_agg = n_series - N
    A_phys = bundle.net.A_phys

    print(f"Data: X={bundle.X.shape}, y_all={bundle.y_all.shape}")
    print(f"Device: {device}")

    # Normalize
    wcfg = WindowConfig()
    split = SplitConfig()
    norm = FeatureNormalizer()
    train_rng, val_rng, test_rng = make_splits(T, split)
    norm.fit(bundle.X[:train_rng[1]])
    X_normed = norm.transform(bundle.X)

    # Load model
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path("research_output/multi_seed/seed_00/checkpoint.pt")

    mcfg = {"d_model": 192, "horizon": 4, "K": 3, "lora_r": 16, "n_heads": 6, "dropout": 0.1}
    model = DTSGSSF(
        N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
        d_model=192, horizon=4, K=3, lora_r=16, dropout=0.1, n_heads=6,
    ).to(device)

    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Using untrained model.")

    model.eval()

    # Feature names for Astana bus network
    feature_names = getattr(bundle, 'feature_names', None)
    if feature_names is None:
        # Default feature names from train_model.py
        feature_names = [
            "passengers_boarding", "passengers_alighting", "load",
            "temperature", "precipitation", "wind_speed",
            "sin_hour", "cos_hour", "sin_dow", "cos_dow",
            "is_weekend", "is_holiday", "is_peak", "route_type"
        ][:F_in]

    # Sample windows from test set for attribution
    ds_test = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])

    # Use 20 random test windows for stable attribution
    n_samples = min(20, len(ds_test))
    indices = np.random.choice(len(ds_test), n_samples, replace=False)

    print(f"Computing Integrated Gradients over {n_samples} test windows ({args.n_steps} steps)...")

    all_attributions = []
    for i, idx in enumerate(indices):
        sample = ds_test[idx]
        x_input = sample["x"].unsqueeze(0).to(device)  # (1, L, N, F_in)
        x_baseline = torch.zeros_like(x_input)  # zero baseline

        attr = compute_integrated_gradients(model, x_baseline, x_input, n_steps=args.n_steps)

        # Average over L and N dimensions to get per-feature importance
        attr_per_feature = attr.abs().mean(dim=(0, 1, 2)).cpu().numpy()  # (F_in,)
        all_attributions.append(attr_per_feature)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_samples} samples processed")

    # Aggregate
    attributions = np.array(all_attributions)  # (n_samples, F_in)
    mean_attr = attributions.mean(axis=0)
    std_attr = attributions.std(axis=0)

    # Normalize to [0, 1]
    mean_attr_norm = mean_attr / mean_attr.sum()

    # Print results
    print("\nFeature Importance (Integrated Gradients):")
    print("-" * 50)
    sorted_idx = np.argsort(mean_attr)[::-1]
    cumulative = 0.0
    for idx in sorted_idx:
        cumulative += mean_attr_norm[idx]
        print(f"  {feature_names[idx]:30s}: {mean_attr_norm[idx]:.4f} ± {std_attr[idx]/mean_attr.sum():.4f}  (cumul: {cumulative:.3f})")

    # Save results
    results = {
        "method": "integrated_gradients",
        "n_steps": args.n_steps,
        "n_samples": n_samples,
        "baseline": "zero",
        "feature_names": feature_names,
        "importance_mean": mean_attr_norm.tolist(),
        "importance_std": (std_attr / mean_attr.sum()).tolist(),
        "raw_mean": mean_attr.tolist(),
        "raw_std": std_attr.tolist(),
    }
    with open(output_dir / "integrated_gradients.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'integrated_gradients.json'}")


if __name__ == "__main__":
    main()
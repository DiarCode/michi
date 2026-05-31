#!/usr/bin/env python3
"""Standalone multi-seed training script for DTS-GSSF.

Runs N independent training seeds with paper-aligned hyperparameters,
computes mean +/- std for all metrics, and saves results to disk.

Usage:
    python experiments/run_training.py --n_seeds 10 --gpu
    python experiments/run_training.py --n_seeds 1 --gpu --epochs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.model import DTSGSSF, nb_nll
from main import (
    load_bundle_pickle, TrainConfig, WindowConfig, SplitConfig,
    set_seed, FeatureNormalizer, WindowDataset, make_splits,
)


def train_one_epoch(model, dl_train, optimizer, device, n_series, lambda_mse=0.3, grad_clip=1.0):
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dl_train:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        optimizer.zero_grad()
        mu, kappa = model(x_batch)
        nll = nb_nll(y_batch, mu, kappa).mean()
        mse = F.mse_loss(mu, y_batch)
        loss = nll + lambda_mse * mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dl_val, device, n_series, horizon=None):
    """Evaluate model, return metrics dict including per-horizon breakdown."""
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0.0
    n_batches = 0

    for batch in dl_val:
        x_batch = batch["x"].to(device)
        y_batch = batch["y"].to(device)
        mu, kappa = model(x_batch)
        nll = nb_nll(y_batch, mu, kappa).mean()
        total_loss += nll.item()
        all_pred.append(mu.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())
        n_batches += 1

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Infer horizon from data if not provided
    if horizon is None:
        horizon = pred.shape[1] if pred.ndim >= 3 else 1

    # Compute overall metrics
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    # MAPE (thresholded)
    mask = np.abs(true) > 5.0
    mape = float(np.mean(np.abs((true[mask] - pred[mask]) / np.abs(true[mask])))) * 100 if mask.sum() > 0 else float('inf')

    result = {
        "val_loss": total_loss / max(n_batches, 1),
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2,
        "val_mape": mape,
    }

    # Per-horizon metrics (pred/true shape: [B, H, n_series])
    if pred.ndim >= 3 and horizon > 1:
        for h in range(horizon):
            pred_h = pred[:, h, :]
            true_h = true[:, h, :]
            mae_h = float(np.mean(np.abs(pred_h - true_h)))
            rmse_h = float(np.sqrt(np.mean((pred_h - true_h) ** 2)))
            ss_res_h = np.sum((true_h - pred_h) ** 2)
            ss_tot_h = np.sum((true_h - np.mean(true_h)) ** 2)
            r2_h = float(1 - ss_res_h / max(ss_tot_h, 1e-8))
            result[f"val_mae_h{h+1}"] = mae_h
            result[f"val_rmse_h{h+1}"] = rmse_h
            result[f"val_r2_h{h+1}"] = r2_h

    return result


def run_seed(seed, bundle, wcfg, split, mcfg, tcfg, device, output_dir):
    """Train and evaluate a single seed."""
    from main import DTSGSSF as MainDTSGSSF
    set_seed(seed)
    print(f"\n[Seed {seed:02d}] Starting training on {device}...")

    T, N, F_in = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    n_agg = n_series - N
    A_phys = bundle.net.A_phys

    # Create model
    model = DTSGSSF(
        N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
        d_model=int(mcfg.get("d_model", 192)),
        horizon=wcfg.horizon,
        K=int(mcfg.get("K", 3)),
        lora_r=int(mcfg.get("lora_r", 16)),
        dropout=float(mcfg.get("dropout", 0.1)),
        n_heads=int(mcfg.get("n_heads", 6)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Seed {seed:02d}] Model: {n_params:,} params")

    # Z-score normalization
    norm = FeatureNormalizer()
    train_rng, val_rng, test_rng = make_splits(T, split)
    norm.fit(bundle.X[:train_rng[1]])
    X_normed = norm.transform(bundle.X)

    # Create datasets
    ds_train = WindowDataset(X_normed, bundle.y_all, wcfg, train_rng[0], train_rng[1])
    ds_val = WindowDataset(X_normed, bundle.y_all, wcfg, val_rng[0], val_rng[1])

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=tcfg.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs, eta_min=1e-6)

    # Training loop
    best_val_r2 = -float('inf')
    best_metrics = None
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_r2": [], "val_mae": [], "val_rmse": []}

    t0 = time.time()
    for epoch in range(tcfg.epochs):
        # Warmup
        if epoch < tcfg.warmup_epochs:
            lr_scale = (epoch + 1) / max(1, tcfg.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = tcfg.lr * lr_scale

        train_loss = train_one_epoch(model, dl_train, optimizer, device, n_series)
        val_metrics = evaluate(model, dl_val, device, n_series, horizon=wcfg.horizon)

        # Record history
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["val_loss"]))
        history["val_r2"].append(float(val_metrics["val_r2"]))
        history["val_mae"].append(float(val_metrics["val_mae"]))
        history["val_rmse"].append(float(val_metrics["val_rmse"]))

        if epoch >= tcfg.warmup_epochs:
            scheduler.step()

        if val_metrics['val_r2'] > best_val_r2:
            best_val_r2 = val_metrics['val_r2']
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Seed {seed:02d}] Epoch {epoch+1:3d}/{tcfg.epochs}: "
                  f"train_loss={train_loss:.4f} val_r2={val_metrics['val_r2']:.4f} "
                  f"val_mae={val_metrics['val_mae']:.4f} best_r2={best_val_r2:.4f}")

        if patience_counter >= tcfg.early_stopping_patience:
            print(f"  [Seed {seed:02d}] Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    ds_test = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=tcfg.batch_size, shuffle=False)
    test_metrics = evaluate(model, dl_test, device, n_series, horizon=wcfg.horizon)

    print(f"[Seed {seed:02d}] Done in {elapsed:.1f}s — "
          f"test_r2={test_metrics.get('val_r2', 0):.4f} "
          f"test_mae={test_metrics.get('val_mae', 0):.4f}")

    # Save results
    seed_dir = output_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "seed": seed,
        "metrics": {k.replace("val_", ""): v for k, v in test_metrics.items()},
        "best_val_r2": best_val_r2,
        "best_val_metrics": {k.replace("val_", ""): v for k, v in best_metrics.items()} if best_metrics else {},
        "history": history,
        "elapsed_seconds": elapsed,
        "n_params": n_params,
    }
    with open(seed_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "normalizer": norm.state_dict(),
        "seed": seed,
        "metrics": results,
        "config": {
            "window": wcfg.__dict__,
            "split": split.__dict__,
            "train": tcfg.__dict__,
            "model": mcfg,
        },
    }, seed_dir / "checkpoint.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed DTS-GSSF training")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--output_dir", type=str, default="research_output/multi_seed")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    bundle = load_bundle_pickle(str(PROJECT_ROOT / "data" / "bundle.pkl"))
    if bundle is None:
        print("ERROR: Could not load bundle.pkl")
        sys.exit(1)

    print(f"Device: {device}")
    print(f"Data: X={bundle.X.shape}, y_all={bundle.y_all.shape}")

    # Config
    wcfg = WindowConfig()
    split = SplitConfig()
    tcfg = TrainConfig(epochs=args.epochs)
    mcfg = {"d_model": 192, "horizon": 4, "K": 3, "lora_r": 16, "n_heads": 6, "dropout": 0.1}

    # Save config
    config = {
        "window": wcfg.__dict__,
        "split": split.__dict__,
        "train": tcfg.__dict__,
        "model": mcfg,
        "n_seeds": args.n_seeds,
        "device": str(device),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run seeds
    all_results = []
    for seed in range(args.n_seeds):
        results = run_seed(seed, bundle, wcfg, split, mcfg, tcfg, device, output_dir)
        all_results.append(results)

    # Aggregate
    from experiments.save_results import aggregate_results, save_aggregate_results
    agg = aggregate_results(all_results)
    agg_path = save_aggregate_results(output_dir, agg)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ({args.n_seeds} seeds)")
    print(f"{'='*60}")
    for key in sorted(agg.keys()):
        if key.startswith("_"):
            continue
        entry = agg[key]
        if isinstance(entry, dict) and "mean" in entry:
            print(f"  {key:30s}: {entry['mean']:.4f} +/- {entry['std']:.4f}")
    print(f"\nResults saved to: {agg_path}")


if __name__ == "__main__":
    main()
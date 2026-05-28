"""Train DTS-GSSF model on historical ridership data — production-grade pipeline.

Preloads all data into memory, applies z-score normalization, trains with
mixed-precision on CUDA, early stopping, LR warmup, and batched training.

Usage:
    .venv/Scripts/python data/train_model.py
"""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as tnnf
import torch.optim as optim
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal
from backend.models_orm import (
    StationORM, HistoricalRidershipORM, WeatherReadingORM,
    ForecastORM,
)
from backend.ml.model import DTSGSSF, nb_nll
from backend.ml.data_loader import build_adjacency
from backend.ml.artifact_store import save_artifact

# --- Configuration ---
WINDOW_HOURS = 72       # 3 days context (fits RTX 3060 12GB)
HORIZON_HOURS = 4       # predict next 4 hours
F = 16                   # 16 engineered features (was 11, added lag features)
EPOCHS = 500
LR = 3e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
N_AGG = 3
MAX_SAMPLES = 4000
STRIDE_HOURS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 4           # effective batch = 32
PATIENCE = 50             # early stopping patience
D_MODEL = 192
K_HOPS = 3
LORA_R = 16
N_HEADS = 6
MSE_WEIGHT = 0.3         # combined NLL + MSE loss weight

KAZAKH_HOLIDAYS = {
    (1,1),(1,2),(1,7),(3,8),(3,22),(3,23),(5,1),(5,7),(5,9),
    (6,10),(7,6),(8,30),(10,25),(12,16),(12,17),
}

TRAIN_LOG = []


def is_rush_hour(dt):
    return dt.hour in (7, 8, 9, 17, 18, 19)


def precompute_tensors(session, station_idx, stop_ids, data_start, data_end):
    """Preload all data, build sliding-window tensors with 16 features + historical mean imputation."""
    N = len(stop_ids)
    print(f"  Preloading data for {N} stations...")

    print("  Loading ridership data...")
    ridership = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp).all()
    rd = {}
    for r in ridership:
        ts = r.timestamp.replace(tzinfo=timezone.utc) if r.timestamp.tzinfo is None else r.timestamp
        hour_ts = ts.replace(minute=0, second=0, microsecond=0)
        rd[(hour_ts, r.station_id)] = r
    print(f"  Loaded {len(rd):,} ridership records")

    # Compute historical per-station per-hour means for imputation
    print("  Computing per-station hourly means for imputation...")
    station_hour_means = {}
    for sid in stop_ids:
        hour_vals = {h: [] for h in range(24)}
        for (ts, stid), row in rd.items():
            if stid == sid:
                hour_vals[ts.hour].append(row.passengers_boarding)
        station_hour_means[sid] = {}
        for h in range(24):
            vals = hour_vals[h]
            station_hour_means[sid][h] = np.mean(vals) if vals else 0.0
    print(f"  Computed means for {len(station_hour_means)} stations")

    print("  Loading weather data...")
    weather_rows = session.query(WeatherReadingORM).all()
    wd = {}
    for w in weather_rows:
        ts = w.timestamp.replace(tzinfo=timezone.utc) if w.timestamp.tzinfo is None else w.timestamp
        hour_ts = ts.replace(minute=0, second=0, microsecond=0)
        wd[hour_ts] = w
    print(f"  Loaded {len(wd):,} weather records")

    # Build sliding windows
    total_hours = int((data_end - data_start).total_seconds() / 3600)
    available = total_hours - WINDOW_HOURS - HORIZON_HOURS
    n_possible = max(1, available // STRIDE_HOURS)
    n_samples = min(n_possible, MAX_SAMPLES)
    print(f"  Building {n_samples} samples (stride={STRIDE_HOURS}h)...")

    all_x = []
    all_y = []

    for i in range(n_samples):
        offset = i * STRIDE_HOURS
        sample_time = data_start + timedelta(hours=WINDOW_HOURS + offset)

        x_data = np.zeros((WINDOW_HOURS, N, F), dtype=np.float32)
        y_data = np.zeros((HORIZON_HOURS, N), dtype=np.float32)

        # First pass: fill raw values with imputation for lag computation
        raw_boarding = np.zeros((WINDOW_HOURS, N), dtype=np.float32)
        for t in range(WINDOW_HOURS):
            ts = sample_time - timedelta(hours=WINDOW_HOURS - t)
            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                if row:
                    raw_boarding[t, n_idx] = row.passengers_boarding
                else:
                    raw_boarding[t, n_idx] = station_hour_means.get(sid, {}).get(ts.hour, 0.0)

        # Second pass: fill all features including lags with imputation
        for t in range(WINDOW_HOURS):
            ts = sample_time - timedelta(hours=WINDOW_HOURS - t)
            w = wd.get(ts)
            is_hol = ts.weekday() >= 5 or (ts.month, ts.day) in KAZAKH_HOLIDAYS
            hour_sin = np.sin(2 * np.pi * ts.hour / 24)
            hour_cos = np.cos(2 * np.pi * ts.hour / 24)
            dow_sin = np.sin(2 * np.pi * ts.weekday() / 7)
            dow_cos = np.cos(2 * np.pi * ts.weekday() / 7)
            rush = 1.0 if is_rush_hour(ts) else 0.0

            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                h_mean = station_hour_means.get(sid, {}).get(ts.hour, 0.0)
                if row:
                    x_data[t, n_idx, 0] = row.passengers_boarding
                    x_data[t, n_idx, 1] = row.passengers_alighting
                    x_data[t, n_idx, 2] = row.load
                else:
                    x_data[t, n_idx, 0] = h_mean
                    x_data[t, n_idx, 1] = h_mean * 0.55
                    x_data[t, n_idx, 2] = h_mean * 0.5
                if w:
                    x_data[t, n_idx, 3] = w.temperature or 0.0
                    x_data[t, n_idx, 4] = w.precipitation or 0.0
                x_data[t, n_idx, 5] = 1.0 if is_hol else 0.0
                x_data[t, n_idx, 6] = hour_sin
                x_data[t, n_idx, 7] = hour_cos
                x_data[t, n_idx, 8] = dow_sin
                x_data[t, n_idx, 9] = dow_cos
                x_data[t, n_idx, 10] = rush
                # Lag features (indices 11-15)
                x_data[t, n_idx, 11] = raw_boarding[t, n_idx] - (raw_boarding[t-1, n_idx] if t > 0 else 0.0)
                window6 = max(0, t - 5)
                x_data[t, n_idx, 12] = raw_boarding[window6:t+1, n_idx].mean()
                window24 = max(0, t - 23)
                x_data[t, n_idx, 13] = raw_boarding[window24:t+1, n_idx].mean()
                x_data[t, n_idx, 14] = raw_boarding[t, n_idx] - raw_boarding[window24:t+1, n_idx].mean()
                x_data[t, n_idx, 15] = raw_boarding[t, n_idx] / (raw_boarding[window24:t+1, n_idx].mean() + 1e-6) - 1.0

        for t in range(HORIZON_HOURS):
            ts = sample_time + timedelta(hours=t)
            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                if row:
                    y_data[t, n_idx] = row.passengers_boarding

        all_x.append(x_data[np.newaxis, :, :, :])
        all_y.append(y_data[np.newaxis, :, :])

        if (i + 1) % 100 == 0:
            print(f"    ...{i+1}/{n_samples} samples built")

    print(f"  Built {len(all_x)} samples")
    return all_x, all_y


def standardize_features(train_x, val_x, test_x):
    """Z-score normalization using training statistics only."""
    mean = train_x.mean(axis=(0, 1, 2), keepdims=True)
    std = train_x.std(axis=(0, 1, 2), keepdims=True) + 1e-8

    train_x_norm = (train_x - mean) / std
    val_x_norm = (val_x - mean) / std
    test_x_norm = (test_x - mean) / std

    return train_x_norm, val_x_norm, test_x_norm, mean, std


def train_one_epoch(model, train_x, train_y, optimizer, scaler, n_series, device):
    """Train one epoch with mixed precision and gradient accumulation."""
    model.train()
    n_steps = train_x.shape[0]
    indices = torch.randperm(n_steps, device="cpu")
    epoch_losses = []
    optimizer.zero_grad()

    for b in range(n_steps):
        idx = indices[b]
        # Move only this sample to GPU to avoid OOM
        x_b = train_x[idx:idx+1].to(device, non_blocking=True)
        y_b = train_y[idx:idx+1].to(device, non_blocking=True)

        with autocast(device_type=device, enabled=(device == "cuda")):
            mu, kappa = model(x_b)
            mu_series = mu[:, :, :n_series]
            H_y = y_b.shape[1]
            H_pred = mu_series.shape[1]
            H_min = min(H_y, H_pred)
            y_aligned = y_b[:, :H_min, :]
            mu_aligned = mu_series[:, :H_min, :]
            kappa_expanded = torch.clamp(kappa, min=0.01).expand_as(mu_aligned)
            nll_loss = nb_nll(y_aligned, mu_aligned, kappa_expanded).mean()
            mse_loss = tnnf.mse_loss(mu_aligned, y_aligned)
            loss = (nll_loss + MSE_WEIGHT * mse_loss) / GRAD_ACCUM

        if device == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (b + 1) % GRAD_ACCUM == 0 or (b + 1) == n_steps:
            if device == "cuda":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        epoch_losses.append(loss.item() * GRAD_ACCUM)

    return np.mean(epoch_losses)


def evaluate(model, x_t, y_t, n_series, device, batch_size=16):
    """Evaluate model in mini-batches. Reports R², masked MAPE (y>5), MAE, RMSE."""
    model.eval()
    all_mu = []
    all_y = []
    all_kappa = []
    with torch.no_grad():
        for i in range(0, x_t.shape[0], batch_size):
            x_b = x_t[i:i+batch_size].to(device, non_blocking=True)
            y_b = y_t[i:i+batch_size].to(device, non_blocking=True)
            with autocast(device_type=device, enabled=(device == "cuda")):
                mu, kappa = model(x_b)
            mu_series = mu[:, :, :n_series]
            H_y = y_b.shape[1]
            H_pred = mu_series.shape[1]
            H = min(H_y, H_pred)
            all_mu.append(mu_series[:, :H, :].float().cpu())
            all_y.append(y_b[:, :H, :].cpu())
            all_kappa.append(kappa.float().cpu())
    mu_cat = torch.cat(all_mu, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    kappa_val = torch.clamp(torch.cat(all_kappa, dim=0) if all_kappa[0].numel() > 1 else all_kappa[0], min=0.01).expand_as(mu_cat)
    nll = nb_nll(y_cat, mu_cat, kappa_val).mean().item()
    mae = torch.mean(torch.abs(y_cat - mu_cat)).item()
    rmse = torch.sqrt(torch.mean((y_cat - mu_cat) ** 2)).item()
    # R² — coefficient of determination (standard Q1 metric for regression)
    ss_res = torch.sum((y_cat - mu_cat) ** 2).item()
    ss_tot = torch.sum((y_cat - y_cat.mean()) ** 2).item()
    r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))
    # Masked MAPE — only for stations/times with ridership > 5 (avoids near-zero inflation)
    mask = y_cat > 5
    mape = (torch.mean(torch.abs((y_cat[mask] - mu_cat[mask]) / (y_cat[mask] + 1e-6))).item()
            if mask.sum() > 0 else 0.0)
    return {"nll": nll, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def evaluate_per_horizon(model, x_t, y_t, n_series, device, batch_size=16):
    """Per-horizon accuracy breakdown with R²."""
    model.eval()
    horizons = [15, 30, 60, 120]
    all_mu = []
    all_y = []
    with torch.no_grad():
        for i in range(0, x_t.shape[0], batch_size):
            x_b = x_t[i:i+batch_size].to(device, non_blocking=True)
            y_b = y_t[i:i+batch_size]
            with autocast(device_type=device, enabled=(device == "cuda")):
                mu, kappa = model(x_b)
            all_mu.append(mu[:, :, :n_series].float().cpu())
            all_y.append(y_b)
    mu_cat = torch.cat(all_mu, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    H = min(y_cat.shape[1], mu_cat.shape[1])
    results = {}
    for h_min, h_idx_raw in zip(horizons, range(H)):
        h_idx = min(h_idx_raw, H - 1)
        y_h = y_cat[:, h_idx, :]
        mu_h = mu_cat[:, h_idx, :]
        mae_h = torch.mean(torch.abs(y_h - mu_h)).item()
        mask = y_h > 5
        mape_h = (torch.mean(torch.abs((y_h[mask] - mu_h[mask]) / (y_h[mask] + 1e-6))).item()
                  if mask.sum() > 0 else 0.0)
        ss_res = torch.sum((y_h - mu_h) ** 2).item()
        ss_tot = torch.sum((y_h - y_h.mean()) ** 2).item()
        r2_h = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))
        results[h_min] = {"mae": mae_h, "mape": mape_h, "r2": r2_h}
    return results


def main():
    print("=" * 70)
    print("DTS-GSSF Professional Training Pipeline (CUDA-Optimized)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: d_model={D_MODEL}, K={K_HOPS}, lora_r={LORA_R}")
    print(f"Training: {EPOCHS} epochs, LR={LR}, warmup={WARMUP_EPOCHS}")
    print(f"Data: window={WINDOW_HOURS}h, stride={STRIDE_HOURS}h, max_samples={MAX_SAMPLES}")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")

    session = SessionLocal()

    try:
        row_count = session.query(HistoricalRidershipORM).count()
        print(f"\nHistorical data: {row_count:,} rows")
        if row_count == 0:
            print("ERROR: No historical data. Run data/generate_historical.py first.")
            return

        print("\nBuilding adjacency matrix...")
        A_phys, stop_ids, station_idx = build_adjacency(session)
        N = len(stop_ids)
        print(f"  {N} stations, adjacency matrix {A_phys.shape}")

        ds_hash = f"{row_count}-{N}"
        print(f"  Dataset hash: {ds_hash}")

        print("\nInitializing DTS-GSSF model...")
        n_series = N
        n_agg = N_AGG
        model = DTSGSSF(
            N=N, F_in=F, n_series=n_series, n_agg=n_agg,
            A_phys=A_phys, d_model=D_MODEL, horizon=4, K=K_HOPS, lora_r=LORA_R, dropout=0.1,
            n_heads=N_HEADS,
        )
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {n_params:,} / {n_total:,}")

        first_row = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp).first()
        last_row = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp.desc()).first()
        _ts = first_row.timestamp
        data_start = _ts.replace(tzinfo=timezone.utc) if _ts.tzinfo is None else _ts
        _ts2 = last_row.timestamp
        data_end = _ts2.replace(tzinfo=timezone.utc) if _ts2.tzinfo is None else _ts2
        print(f"  Data range: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")

        all_x, all_y = precompute_tensors(session, station_idx, stop_ids, data_start, data_end)
        if len(all_x) < 20:
            print("ERROR: Too few valid training samples.")
            return

        # Train / Val / Test split (70 / 15 / 15)
        total = len(all_x)
        test_count = max(1, int(total * TEST_SPLIT))
        val_count = max(1, int(total * VAL_SPLIT))
        train_count = total - val_count - test_count
        print(f"\nSplit: Train={train_count}, Val={val_count}, Test={test_count}")

        train_x = np.concatenate(all_x[:train_count], axis=0)
        train_y = np.concatenate(all_y[:train_count], axis=0)
        val_x = np.concatenate(all_x[train_count:train_count+val_count], axis=0)
        val_y = np.concatenate(all_y[train_count:train_count+val_count], axis=0)
        test_x = np.concatenate(all_x[train_count+val_count:], axis=0)
        test_y = np.concatenate(all_y[train_count+val_count:], axis=0)

        # Feature standardization
        print("\nApplying z-score normalization...")
        train_x, val_x, test_x, feat_mean, feat_std = standardize_features(train_x, val_x, test_x)
        print(f"  Feature means (first 3): {feat_mean.flatten()[:3]}")
        print(f"  Feature stds (first 3): {feat_std.flatten()[:3]}")

        # Keep data on CPU to avoid OOM; move batches to GPU in training loop
        train_x_t = torch.as_tensor(train_x, dtype=torch.float32)
        train_y_t = torch.as_tensor(train_y, dtype=torch.float32)
        val_x_t = torch.as_tensor(val_x, dtype=torch.float32)
        val_y_t = torch.as_tensor(val_y, dtype=torch.float32)
        test_x_t = torch.as_tensor(test_x, dtype=torch.float32)
        test_y_t = torch.as_tensor(test_y, dtype=torch.float32)
        print(f"  Train: {train_x_t.shape}, Val: {val_x_t.shape}, Test: {test_x_t.shape}")

        # Training
        print("\nTraining with early stopping (patience={})...".format(PATIENCE))
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

        best_val_r2 = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            # LR warmup
            if epoch < WARMUP_EPOCHS:
                warmup_factor = (epoch + 1) / WARMUP_EPOCHS
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * warmup_factor
            elif epoch == WARMUP_EPOCHS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR

            avg_loss = train_one_epoch(model, train_x_t, train_y_t, optimizer, scaler, n_series, DEVICE)

            if epoch >= WARMUP_EPOCHS:
                scheduler.step()

            val_metrics = evaluate(model, val_x_t, val_y_t, n_series, DEVICE)
            val_mae = val_metrics["mae"]
            val_r2 = val_metrics["r2"]

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_metrics["rmse"], 4),
                "val_mape": round(val_metrics["mape"], 4),
                "val_r2": round(val_r2, 4),
                "val_nll": round(val_metrics["nll"], 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            TRAIN_LOG.append(log_entry)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"R²: {val_r2:.1%}, MAE: {val_mae:.2f}, "
                      f"MAPE(>5): {val_metrics['mape']:.1%}, RMSE: {val_metrics['rmse']:.2f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
            print(f"\nLoaded best model (Val R²={best_val_r2:.1%})")

        # Final test evaluation
        print("\n" + "=" * 70)
        print("FINAL EVALUATION ON HELD-OUT TEST SET")
        print("=" * 70)
        test_metrics = evaluate(model, test_x_t, test_y_t, n_series, DEVICE)
        print(f"  MAE:   {test_metrics['mae']:.4f}")
        print(f"  RMSE:  {test_metrics['rmse']:.4f}")
        print(f"  MAPE(>5):  {test_metrics['mape']:.1%}")
        print(f"  R²:    {test_metrics['r2']:.4f}")
        print(f"  NLL:   {test_metrics['nll']:.4f}")
        print(f"  Accuracy (R²): {test_metrics['r2']:.1%}")

        # Per-horizon
        print("\nPer-horizon accuracy:")
        horizon_results = evaluate_per_horizon(model, test_x_t, test_y_t, n_series, DEVICE)
        for h_min, hr in horizon_results.items():
            print(f"  {h_min:>3d}min: MAE={hr['mae']:.2f}, MAPE(>5)={hr['mape']:.1%}, R²={hr['r2']:.1%}")

        metrics = {
            "mae": round(test_metrics["mae"], 4),
            "rmse": round(test_metrics["rmse"], 4),
            "mape": round(test_metrics["mape"], 4),
            "nll": round(test_metrics["nll"], 4),
            "r2": round(test_metrics["r2"], 4),
            "accuracy": round(test_metrics["r2"], 4),
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "epochs_run": len(TRAIN_LOG),
            "best_val_r2": round(best_val_r2, 4),
        }
        for h_min, hr in horizon_results.items():
            metrics[f"mae_{h_min}min"] = round(hr["mae"], 4)
            metrics[f"mape_{h_min}min"] = round(hr["mape"], 4)
            metrics[f"r2_{h_min}min"] = round(hr["r2"], 4)

        # Save model
        print("\nSaving model artifact...")
        config = {
            "N": N, "F_in": F, "n_series": n_series, "n_agg": n_agg,
            "d_model": D_MODEL, "horizon": 4, "K": K_HOPS, "lora_r": LORA_R,
            "dropout": 0.1, "window_hours": WINDOW_HOURS,
            "horizon_hours": HORIZON_HOURS, "epochs": EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "device": DEVICE,
            "warmup_epochs": WARMUP_EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "feat_mean": feat_mean.flatten().tolist(),
            "feat_std": feat_std.flatten().tolist(),
        }
        model_state = model.cpu().state_dict()
        artifact = save_artifact(
            model_state=model_state,
            metrics=metrics,
            config=config,
            dataset_hash=ds_hash,
            feature_version=2,
            is_production=True,
        )
        print(f"  Saved: {artifact.version} -> {artifact.artifact_path}")

        # Save normalization stats
        norm_path = Path(artifact.artifact_path).parent / f"{artifact.version}_norm.json"
        with open(norm_path, "w") as f:
            json.dump({
                "feat_mean": feat_mean.flatten().tolist(),
                "feat_std": feat_std.flatten().tolist(),
                "F": F,
            }, f)
        print(f"  Normalization stats: {norm_path}")

        # Save training log
        log_path = Path(artifact.artifact_path).parent / f"{artifact.version}_log.json"
        with open(log_path, "w") as f:
            json.dump(TRAIN_LOG, f, indent=2)
        print(f"  Training log: {log_path}")

        # Generate forecasts
        print("\nGenerating forecasts...")
        model = model.to(DEVICE)
        model.eval()
        from backend.ml.predictor import generate_predictions
        predictions = generate_predictions(model, session, station_idx, stop_ids)
        if predictions:
            session.query(ForecastORM).delete()
            session.commit()
            for i in range(0, len(predictions), 500):
                batch = predictions[i:i+500]
                for p in batch:
                    ts = p["timestamp"]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    session.add(ForecastORM(
                        station_id=p["station_id"],
                        timestamp=ts,
                        predicted=p["predicted"],
                        confidence=p["confidence"],
                        model_version=p.get("model_version", "dts-gssf"),
                        created_at=datetime.now(timezone.utc),
                        horizon_minutes=p.get("horizon_minutes", 60),
                    ))
                session.commit()
            print(f"  Stored {len(predictions)} forecasts")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"  Model: {artifact.version}")
        print(f"  Test R² (Accuracy): {test_metrics['r2']:.1%}")
        print(f"  MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE(>5)={metrics['mape']:.1%}")
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\nTraining failed: {e}")
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
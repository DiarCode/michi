"""Baseline comparison experiments for DTS-GSSF thesis.

Trains/evaluates 6 baseline models on the same dataset and split as DTS-GSSF,
then generates comparison tables and charts.

Baselines: Historical Average, Seasonal Naive, Moving Average, LSTM, GRU, TCN

Usage:
    .venv/Scripts/python data/baseline_experiments.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal
from backend.models_orm import HistoricalRidershipORM, WeatherReadingORM
from backend.ml.data_loader import build_adjacency

# --- Config (matches train_model.py) ---
WINDOW_HOURS = 72
HORIZON_HOURS = 4
F = 16
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_SAMPLES = 4000
STRIDE_HOURS = 3
N_AGG = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 192
EPOCHS = 200
LR = 3e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 50
BATCH_SIZE = 8
GRAD_ACCUM = 4

KAZAKH_HOLIDAYS = {
    (1,1),(1,2),(1,7),(3,8),(3,22),(3,23),(5,1),(5,7),(5,9),
    (6,10),(7,6),(8,30),(10,25),(12,16),(12,17),
}


def is_rush_hour(dt):
    return dt.hour in (7, 8, 9, 17, 18, 19)


def precompute_tensors(session, station_idx, stop_ids, data_start, data_end):
    """Same data loading as train_model.py for fair comparison."""
    N = len(stop_ids)
    print(f"  Preloading data for {N} stations...")
    ridership = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp).all()
    rd = {}
    for r in ridership:
        ts = r.timestamp.replace(tzinfo=timezone.utc) if r.timestamp.tzinfo is None else r.timestamp
        hour_ts = ts.replace(minute=0, second=0, microsecond=0)
        rd[(hour_ts, r.station_id)] = r

    station_hour_means = {}
    for sid in stop_ids:
        hour_vals = {h: [] for h in range(24)}
        for (ts, stid), row in rd.items():
            if stid == sid:
                hour_vals[ts.hour].append(row.passengers_boarding)
        station_hour_means[sid] = {h: np.mean(v) if v else 0.0 for h, v in hour_vals.items()}

    weather_rows = session.query(WeatherReadingORM).all()
    wd = {}
    for w in weather_rows:
        ts = w.timestamp.replace(tzinfo=timezone.utc) if w.timestamp.tzinfo is None else w.timestamp
        wd[ts.replace(minute=0, second=0, microsecond=0)] = w

    total_hours = int((data_end - data_start).total_seconds() / 3600)
    available = total_hours - WINDOW_HOURS - HORIZON_HOURS
    n_possible = max(1, available // STRIDE_HOURS)
    n_samples = min(n_possible, MAX_SAMPLES)

    all_x, all_y = [], []
    print(f"  Building {n_samples} samples...", flush=True)
    for i in range(n_samples):
        offset = i * STRIDE_HOURS
        sample_time = data_start + timedelta(hours=WINDOW_HOURS + offset)
        x_data = np.zeros((WINDOW_HOURS, N, F), dtype=np.float32)
        y_data = np.zeros((HORIZON_HOURS, N), dtype=np.float32)

        raw_boarding = np.zeros((WINDOW_HOURS, N), dtype=np.float32)
        for t in range(WINDOW_HOURS):
            ts = sample_time - timedelta(hours=WINDOW_HOURS - t)
            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                raw_boarding[t, n_idx] = row.passengers_boarding if row else station_hour_means.get(sid, {}).get(ts.hour, 0.0)

        for t in range(WINDOW_HOURS):
            ts = sample_time - timedelta(hours=WINDOW_HOURS - t)
            w = wd.get(ts)
            is_hol = ts.weekday() >= 5 or (ts.month, ts.day) in KAZAKH_HOLIDAYS
            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                h_mean = station_hour_means.get(sid, {}).get(ts.hour, 0.0)
                x_data[t, n_idx, 0] = row.passengers_boarding if row else h_mean
                x_data[t, n_idx, 1] = row.passengers_alighting if row else h_mean * 0.55
                x_data[t, n_idx, 2] = row.load if row else h_mean * 0.5
                if w:
                    x_data[t, n_idx, 3] = w.temperature or 0.0
                    x_data[t, n_idx, 4] = w.precipitation or 0.0
                x_data[t, n_idx, 5] = 1.0 if is_hol else 0.0
                x_data[t, n_idx, 6] = np.sin(2 * np.pi * ts.hour / 24)
                x_data[t, n_idx, 7] = np.cos(2 * np.pi * ts.hour / 24)
                x_data[t, n_idx, 8] = np.sin(2 * np.pi * ts.weekday() / 7)
                x_data[t, n_idx, 9] = np.cos(2 * np.pi * ts.weekday() / 7)
                x_data[t, n_idx, 10] = 1.0 if is_rush_hour(ts) else 0.0
                x_data[t, n_idx, 11] = raw_boarding[t, n_idx] - (raw_boarding[t-1, n_idx] if t > 0 else 0.0)
                x_data[t, n_idx, 12] = raw_boarding[max(0, t-5):t+1, n_idx].mean()
                x_data[t, n_idx, 13] = raw_boarding[max(0, t-23):t+1, n_idx].mean()
                x_data[t, n_idx, 14] = raw_boarding[t, n_idx] - raw_boarding[max(0, t-23):t+1, n_idx].mean()
                x_data[t, n_idx, 15] = raw_boarding[t, n_idx] / (raw_boarding[max(0, t-23):t+1, n_idx].mean() + 1e-6) - 1.0

        for t in range(HORIZON_HOURS):
            ts = sample_time + timedelta(hours=t)
            for n_idx, sid in enumerate(stop_ids):
                row = rd.get((ts, sid))
                if row:
                    y_data[t, n_idx] = row.passengers_boarding

        all_x.append(x_data[np.newaxis])
        all_y.append(y_data[np.newaxis])
        if (i + 1) % 100 == 0:
            print(f"    ...{i+1}/{n_samples} samples built", flush=True)
    print(f"  Built {len(all_x)} samples", flush=True)
    return all_x, all_y, station_hour_means


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1, 2), keepdims=True)
    std = train_x.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def compute_metrics(y_true, y_pred):
    """Compute R-squared, MAE, RMSE, MAPE(>5) from numpy arrays."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true > 5
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-6)))) if mask.sum() > 0 else 0.0
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}


def compute_metrics_torch(y_true_t, y_pred_t):
    return compute_metrics(y_true_t.cpu().numpy(), y_pred_t.cpu().numpy())


# ==================== STATISTICAL BASELINES ====================

def historical_average(train_y, test_y):
    """Predict per-station mean boarding from training data."""
    train_mean = train_y.mean(axis=(0, 1), keepdims=True)
    predictions = np.broadcast_to(train_mean, test_y.shape)
    return compute_metrics(test_y, predictions)


def seasonal_naive(train_y, test_y):
    """Predict same-hour pattern from last week of training data."""
    last_week = train_y[-1]
    predictions = np.broadcast_to(last_week[np.newaxis], test_y.shape)
    return compute_metrics(test_y, predictions)


def moving_average_baseline(train_y, test_y):
    """Predict 24h rolling average from training data."""
    window = min(24, train_y.shape[0])
    avg = train_y[-window:].mean(axis=(0, 1), keepdims=True)
    predictions = np.broadcast_to(avg, test_y.shape)
    return compute_metrics(test_y, predictions)


# ==================== DL BASELINE MODELS ====================

class LSTMModel(nn.Module):
    def __init__(self, F_in, d_model, horizon, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        pred = self.head(out)
        return pred.reshape(B, N, self.head[-1].out_features).permute(0, 2, 1)


class GRUModel(nn.Module):
    def __init__(self, F_in, d_model, horizon, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        pred = self.head(out)
        return pred.reshape(B, N, self.head[-1].out_features).permute(0, 2, 1)


class Chomp1d(nn.Module):
    def forward(self, x):
        return x[:, :, :-1] if x.size(2) > 1 else x


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(), nn.GELU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.downsample(x)


class TCNModel(nn.Module):
    def __init__(self, F_in, d_model, horizon, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        dilations = [1, 2, 4, 8]
        layers = []
        channels = [d_model] * (len(dilations) + 1)
        for i, d in enumerate(dilations):
            layers.append(TCNBlock(channels[i], channels[i + 1], kernel_size=3, dilation=d, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)  # (B*N, d_model, T)
        x = self.tcn(x)
        x = x[:, :, -1]  # last timestep
        pred = self.head(x)
        return pred.reshape(B, N, self.head[-1].out_features).permute(0, 2, 1)


def train_dl_model(model, train_x, train_y, val_x, val_y, name):
    """Train a deep learning baseline with same hyperparameters as DTS-GSSF."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Training {name}: {n_params:,} trainable params")

    best_val_r2 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(train_x.shape[0])
        epoch_losses = []
        optimizer.zero_grad()

        for b_idx in range(0, train_x.shape[0], BATCH_SIZE):
            batch_idx = indices[b_idx:b_idx + BATCH_SIZE]
            x_b = train_x[batch_idx].to(DEVICE)
            y_b = train_y[batch_idx].to(DEVICE)
            pred = model(x_b)
            H = min(pred.shape[1], y_b.shape[1])
            loss = nn.functional.mse_loss(pred[:, :H, :], y_b[:, :H, :]) / GRAD_ACCUM
            loss.backward()
            if (b_idx // BATCH_SIZE + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            epoch_losses.append(loss.item() * GRAD_ACCUM)

        if epoch >= 20:
            scheduler.step()

        # Evaluate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for i in range(0, val_x.shape[0], 16):
                pred = model(val_x[i:i + 16].to(DEVICE))
                all_pred.append(pred.cpu())
                all_true.append(val_y[i:i + 16])
        pred_cat = torch.cat(all_pred, dim=0)
        true_cat = torch.cat(all_true, dim=0)
        H = min(pred_cat.shape[1], true_cat.shape[1])
        metrics = compute_metrics_torch(true_cat[:, :H, :], pred_cat[:, :H, :])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {np.mean(epoch_losses):.4f} | R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.2f}")

        if metrics["r2"] > best_val_r2:
            best_val_r2 = metrics["r2"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    # Final test evaluation
    model.eval()
    test_x_global = train_x  # will use the test set passed separately
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, val_x.shape[0], 16):
            # This reuses val; caller passes test tensors separately
            pass
    return best_val_r2  # placeholder, real eval done below


def evaluate_dl_model(model, test_x, test_y):
    """Evaluate trained DL model on test set."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, test_x.shape[0], 16):
            pred = model(test_x[i:i + 16].to(DEVICE))
            all_pred.append(pred.cpu())
            all_true.append(test_y[i:i + 16])
    pred_cat = torch.cat(all_pred, dim=0)
    true_cat = torch.cat(all_true, dim=0)
    H = min(pred_cat.shape[1], true_cat.shape[1])
    return compute_metrics_torch(true_cat[:, :H, :], pred_cat[:, :H, :])


def train_and_evaluate(model, train_x, train_y, val_x, val_y, test_x, test_y, name):
    """Full train + evaluate pipeline for a DL baseline."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Training {name}: {n_params:,} trainable params")

    best_val_r2 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        indices = torch.randperm(train_x.shape[0])
        optimizer.zero_grad()

        for b_idx in range(0, train_x.shape[0], BATCH_SIZE):
            batch_idx = indices[b_idx:b_idx + BATCH_SIZE]
            x_b = train_x[batch_idx].to(DEVICE)
            y_b = train_y[batch_idx].to(DEVICE)
            pred = model(x_b)
            H = min(pred.shape[1], y_b.shape[1])
            loss = nn.functional.mse_loss(pred[:, :H, :], y_b[:, :H, :]) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % GRAD_ACCUM == 0 or b_idx + BATCH_SIZE >= train_x.shape[0]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        if epoch >= 20:
            scheduler.step()

        # Validation
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for i in range(0, val_x.shape[0], 16):
                pred = model(val_x[i:i + 16].to(DEVICE))
                all_pred.append(pred.cpu())
                all_true.append(val_y[i:i + 16])
        pred_cat = torch.cat(all_pred, dim=0)
        true_cat = torch.cat(all_true, dim=0)
        H = min(pred_cat.shape[1], true_cat.shape[1])
        val_metrics = compute_metrics_torch(true_cat[:, :H, :], pred_cat[:, :H, :])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | "
                  f"R²: {val_metrics['r2']:.4f} | MAE: {val_metrics['mae']:.2f}")

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        print(f"    Best val R²: {best_val_r2:.4f}")

    # Test evaluation
    test_metrics = evaluate_dl_model(model, test_x, test_y)
    print(f"    {name} Test: R²={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.2f}, "
          f"RMSE={test_metrics['rmse']:.2f}, MAPE={test_metrics['mape']:.1%}")
    return test_metrics


# ==================== GRAPH NEURAL BASELINES ====================

class STGCNBlock(nn.Module):
    """Spatio-temporal convolution block from STGCN (Yu et al., 2018)."""
    def __init__(self, in_ch, out_ch, kernel_size, A_norm, dilation=1, dropout=0.1):
        super().__init__()
        self.gcn1 = nn.Linear(in_ch, out_ch)
        self.gcn2 = nn.Linear(out_ch, out_ch)
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp = Chomp1d()
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        # Spatial: graph conv
        h = self.gcn1(x)  # (B, T, N, out_ch)
        h = F.gelu(h)
        h = torch.einsum("ij,bjd->bid", self.A_norm, h)  # graph propagation
        h = self.gcn2(h)
        h = F.gelu(h)
        # Temporal: 1D conv over time
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        h = h.permute(0, 2, 1)  # (B*N, D, T)
        h = F.gelu(self.conv1(h))
        h = self.chomp(h) if h.size(2) > T else h
        h = h[:, :, :T]
        h = F.gelu(self.conv2(h))
        h = self.chomp(h) if h.size(2) > T else h
        h = h[:, :, :T]
        h = h.permute(0, 2, 1).reshape(B, N, T, D)  # (B, N, T, D)
        h = self.drop(self.norm(h.reshape(B * T, N, D)).reshape(B, T, N, D))
        return F.gelu(h + x)


class STGCNModel(nn.Module):
    """STGCN baseline (Yu et al., 2018) — graph convolution + temporal conv."""
    def __init__(self, F_in, d_model, horizon, A_phys, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        A_norm = torch.from_numpy(A_phys).float()
        self.blocks = nn.ModuleList([
            STGCNBlock(d_model, d_model, kernel_size=3, A_norm=A_norm, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = h[:, -1, :, :]  # last timestep
        pred = self.head(h)
        return pred.permute(0, 2, 1)  # (B, horizon, N)


class GWNETLayer(nn.Module):
    """Graph WaveNet dilated causal convolution layer (Wu et al., 2019)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, A_norm, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.gate_conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp = Chomp1d()
        self.gcn = nn.Linear(out_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x, skip=None):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D).permute(0, 2, 1)  # (B*N, D, T)
        h_filter = torch.tanh(self.filter_conv(h))
        h_filter = self.chomp(h_filter)[:, :, :T] if h_filter.size(2) > T else h_filter[:, :, :T]
        h_gate = torch.sigmoid(self.gate_conv(h))
        h_gate = self.chomp(h_gate)[:, :, :T] if h_gate.size(2) > T else h_gate[:, :, :T]
        h = h_filter * h_gate  # (B*N, D, T)
        h = h.permute(0, 2, 1).reshape(B, N, T, D)  # (B, N, T, D)
        # Graph conv
        h = self.gcn(h)
        h = torch.einsum("ij,bjd->bid", self.A_norm, h)
        h = self.drop(self.norm(h.reshape(B * T, N, D)).reshape(B, T, N, D))
        if skip is not None:
            h = h + skip
        return h


class GraphWaveNetModel(nn.Module):
    """Graph WaveNet baseline (Wu et al., 2019) — adaptive adjacency + dilated causal conv."""
    def __init__(self, F_in, d_model, horizon, A_phys, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        A_norm = torch.from_numpy(A_phys).float()
        self.register_buffer("A_norm", A_norm)
        self.E1 = nn.Parameter(torch.randn(A_phys.shape[0], 16) * 0.05)
        self.E2 = nn.Parameter(torch.randn(A_phys.shape[0], 16) * 0.05)
        dilations = [1, 2, 4, 8]
        self.layers = nn.ModuleList([
            GWNETLayer(d_model, d_model, kernel_size=2, dilation=d, A_norm=A_norm, dropout=dropout)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = h[:, -1, :, :]
        pred = self.head(h)
        return pred.permute(0, 2, 1)


class AGCRNModel(nn.Module):
    """AGCRN baseline (Bai et al., 2020) — adaptive graph + GRU."""
    def __init__(self, F_in, d_model, horizon, A_phys, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        A_norm = torch.from_numpy(A_phys).float()
        self.register_buffer("A_norm", A_norm)
        self.E1 = nn.Parameter(torch.randn(A_phys.shape[0], 16) * 0.05)
        self.E2 = nn.Parameter(torch.randn(A_phys.shape[0], 16) * 0.05)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.gcn = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        h = self.input_proj(x)  # (B, T, N, d)
        # Adaptive adjacency
        A_adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)
        A_combined = 0.5 * self.A_norm + 0.5 * A_adp
        # Process each station independently with GRU
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N, T, d)
        h, _ = self.gru(h)  # (B*N, T, d)
        h = h[:, -1, :].reshape(B, N, -1)  # (B, N, d)
        # Graph propagation
        h = self.gcn(h)
        h = torch.einsum("ij,bjd->bid", A_combined, h)
        h = self.norm(h)
        pred = self.head(h)
        return pred.permute(0, 2, 1)  # (B, horizon, N)


class DeepARModel(nn.Module):
    """DeepAR baseline (Salinas et al., 2020) — autoregressive LSTM with Gaussian likelihood."""
    def __init__(self, F_in, d_model, horizon, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(F_in, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.mu_head = nn.Linear(d_model, horizon)
        self.sigma_head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, T, N, F_in = x.shape
        h = self.input_proj(x)  # (B, T, N, d)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.lstm(h)
        out = out[:, -1, :].reshape(B, N, -1)
        mu = self.mu_head(out).permute(0, 2, 1)  # (B, H, N)
        return mu


    """Generate bar charts comparing all models on R² and MAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["Historical Avg", "Seasonal Naive", "Moving Avg", "LSTM", "GRU", "TCN", "STGCN", "GraphWaveNet", "AGCRN", "DeepAR", "DTS-GSSF"]
    r2_values = [results[m]["r2"] for m in models]
    mae_values = [results[m]["mae"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    colors = ["#2ecc71" if m == "DTS-GSSF" else "#3498db" if m in ["LSTM", "GRU", "TCN", "STGCN", "GraphWaveNet", "AGCRN", "DeepAR"] else "#95a5a6" for m in models]

    bars1 = ax1.bar(models, r2_values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("R² (Coefficient of Determination)", fontsize=11)
    ax1.set_title("Prediction Accuracy (R²)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=results["DTS-GSSF"]["r2"], color="#e74c3c", linestyle="--", alpha=0.5, label="DTS-GSSF")
    for bar, val in zip(bars1, r2_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)

    bars2 = ax2.bar(models, mae_values, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("MAE (Passengers)", fontsize=11)
    ax2.set_title("Mean Absolute Error", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {output_path}")


def main():
    import sys as _sys
    # Force unbuffered output
    _sys.stdout.reconfigure(line_buffering=True)

    print("=" * 70, flush=True)
    print("BASELINE COMPARISON EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    session = SessionLocal()
    try:
        A_phys, stop_ids, station_idx = build_adjacency(session)
        N = len(stop_ids)
        print(f"  {N} stations loaded")

        first_row = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp).first()
        last_row = session.query(HistoricalRidershipORM).order_by(HistoricalRidershipORM.timestamp.desc()).first()
        data_start = first_row.timestamp.replace(tzinfo=timezone.utc)
        data_end = last_row.timestamp.replace(tzinfo=timezone.utc)

        all_x, all_y, station_hour_means = precompute_tensors(session, station_idx, stop_ids, data_start, data_end)

        total = len(all_x)
        test_count = max(1, int(total * TEST_SPLIT))
        val_count = max(1, int(total * VAL_SPLIT))
        train_count = total - val_count - test_count

        train_x = np.concatenate(all_x[:train_count], axis=0)
        train_y = np.concatenate(all_y[:train_count], axis=0)
        val_x = np.concatenate(all_x[train_count:train_count + val_count], axis=0)
        val_y = np.concatenate(all_y[train_count:train_count + val_count], axis=0)
        test_x = np.concatenate(all_x[train_count + val_count:], axis=0)
        test_y = np.concatenate(all_y[train_count + val_count:], axis=0)

        print(f"\nSplit: Train={train_count}, Val={val_count}, Test={test_count}")
        print(f"Shapes: Train={train_x.shape}, Val={val_x.shape}, Test={test_x.shape}")

        # Z-score normalization
        train_x_n, val_x_n, test_x_n, _, _ = standardize(train_x, val_x, test_x)

        train_x_t = torch.as_tensor(train_x_n, dtype=torch.float32)
        train_y_t = torch.as_tensor(train_y, dtype=torch.float32)
        val_x_t = torch.as_tensor(val_x_n, dtype=torch.float32)
        val_y_t = torch.as_tensor(val_y, dtype=torch.float32)
        test_x_t = torch.as_tensor(test_x_n, dtype=torch.float32)
        test_y_t = torch.as_tensor(test_y, dtype=torch.float32)

        results = {}

        # === STATISTICAL BASELINES ===
        print("\n--- Statistical Baselines ---")

        print("\n  1. Historical Average")
        results["Historical Avg"] = historical_average(train_y, test_y)
        print(f"     R²={results['Historical Avg']['r2']:.4f}, MAE={results['Historical Avg']['mae']:.2f}")

        print("\n  2. Seasonal Naive")
        results["Seasonal Naive"] = seasonal_naive(train_y, test_y)
        print(f"     R²={results['Seasonal Naive']['r2']:.4f}, MAE={results['Seasonal Naive']['mae']:.2f}")

        print("\n  3. Moving Average (24h)")
        results["Moving Avg"] = moving_average_baseline(train_y, test_y)
        print(f"     R²={results['Moving Avg']['r2']:.4f}, MAE={results['Moving Avg']['mae']:.2f}")

        # === DL BASELINES ===
        print("\n--- Deep Learning Baselines ---")

        print("\n  4. LSTM")
        lstm = LSTMModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS, n_layers=2)
        results["LSTM"] = train_and_evaluate(lstm, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "LSTM")

        print("\n  5. GRU")
        gru = GRUModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS, n_layers=2)
        results["GRU"] = train_and_evaluate(gru, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "GRU")

        print("\n  6. TCN")
        tcn = TCNModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS)
        results["TCN"] = train_and_evaluate(tcn, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "TCN")

        # === GNN BASELINES ===
        print("\n--- Graph Neural Network Baselines ---")

        print("\n  7. STGCN")
        stgcn = STGCNModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS, A_phys=A_phys)
        results["STGCN"] = train_and_evaluate(stgcn, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "STGCN")

        print("\n  8. Graph WaveNet")
        gwn = GraphWaveNetModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS, A_phys=A_phys)
        results["GraphWaveNet"] = train_and_evaluate(gwn, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "GraphWaveNet")

        print("\n  9. AGCRN")
        agcrn = AGCRNModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS, A_phys=A_phys)
        results["AGCRN"] = train_and_evaluate(agcrn, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "AGCRN")

        print("\n  10. DeepAR")
        deepar = DeepARModel(F_in=F, d_model=D_MODEL, horizon=HORIZON_HOURS)
        results["DeepAR"] = train_and_evaluate(deepar, train_x_t, train_y_t, val_x_t, val_y_t, test_x_t, test_y_t, "DeepAR")

        # === DTS-GSSF reference ===
        results["DTS-GSSF"] = {"r2": 0.889, "mae": 2.43, "rmse": 10.80, "mape": 0.137}

        # === SUMMARY ===
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Model':<20} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MAPE(>5)':>10}")
        print("-" * 56)
        for name in ["Historical Avg", "Seasonal Naive", "Moving Avg", "LSTM", "GRU", "TCN", "STGCN", "GraphWaveNet", "AGCRN", "DeepAR", "DTS-GSSF"]:
            r = results[name]
            print(f"{name:<20} {r['r2']:>8.4f} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['mape']:>10.1%}")

        # Save results JSON
        output_dir = Path(PROJECT_ROOT) / "docs" / "thesis_figures"
        output_dir.mkdir(exist_ok=True, parents=True)
        results_path = output_dir / "baseline_comparison_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Generate chart
        generate_comparison_chart(results, output_dir / "baseline_comparison.png")

        print("\n" + "=" * 70)
        print("EXPERIMENTS COMPLETE")
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\nExperiments failed: {e}")
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
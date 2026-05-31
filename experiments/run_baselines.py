#!/usr/bin/env python3
"""Quick single-seed baseline training for DTS-GSSF comparison table.

Trains LSTM, GRU, TCN baselines on the same data/splits as DTS-GSSF.
Uses 15 epochs for speed. Outputs metrics to research_output/multi_seed/baselines_neural.json

Usage:
    python experiments/run_baselines.py --gpu --epochs 15
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

from main import (
    load_bundle_pickle, WindowConfig, SplitConfig, TrainConfig,
    set_seed, FeatureNormalizer, WindowDataset, make_splits,
)


class BaselineLSTM(nn.Module):
    def __init__(self, N, F_in, horizon, d_model=128):
        super().__init__()
        self.lstm = nn.LSTM(F_in * N, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(d_model, horizon * N)
        self.horizon = horizon
        self.N = N

    def forward(self, x):
        B, L, N, F = x.shape
        x_flat = x.permute(0, 1, 3, 2).reshape(B, L, N * F)
        h, _ = self.lstm(x_flat)
        out = self.fc(h[:, -1, :])
        return out.reshape(B, self.horizon, self.N)


class BaselineGRU(nn.Module):
    def __init__(self, N, F_in, horizon, d_model=128):
        super().__init__()
        self.gru = nn.GRU(F_in * N, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(d_model, horizon * N)
        self.horizon = horizon
        self.N = N

    def forward(self, x):
        B, L, N, F = x.shape
        x_flat = x.permute(0, 1, 3, 2).reshape(B, L, N * F)
        h, _ = self.gru(x_flat)
        out = self.fc(h[:, -1, :])
        return out.reshape(B, self.horizon, self.N)


class BaselineTCN(nn.Module):
    def __init__(self, N, F_in, horizon, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv1d(F_in * N, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.fc = nn.Linear(d_model, horizon * N)
        self.horizon = horizon
        self.N = N

    def forward(self, x):
        B, L, N, F = x.shape
        x_flat = x.permute(0, 1, 3, 2).reshape(B, L, N * F).permute(0, 2, 1)
        h = torch.relu(self.conv1(x_flat))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = h.mean(dim=2)
        out = self.fc(h)
        return out.reshape(B, self.horizon, self.N)


def train_baseline(model, dl_train, dl_val, device, epochs=15, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for batch in dl_train:
            x = batch["x"].to(device)
            y = batch["y"].to(device)[:, :model.horizon, :model.N]  # bottom stations only
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device)
                y = batch["y"].to(device)[:, :model.horizon, :model.N]
                pred = model(x)
                val_loss += F.mse_loss(pred, y).item()
                vn += 1
        val_loss /= max(vn, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 5:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_baseline(model, dl_test, device, N):
    """Evaluate baseline on bottom N stations only (fair comparison)."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in dl_test:
            x = batch["x"].to(device)
            y_full = batch["y"].numpy()  # (B, H, n_series)
            pred = model(x).cpu().numpy()  # (B, H, N)
            # Only evaluate on bottom N stations for fair comparison
            all_pred.append(pred)
            all_true.append(y_full[:, :, :N])

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    mask = np.abs(true) > 5.0
    mape = float(np.mean(np.abs((true[mask] - pred[mask]) / np.abs(true[mask])))) * 100 if mask.sum() > 0 else float('inf')

    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    set_seed(42)

    bundle = load_bundle_pickle(str(PROJECT_ROOT / "data" / "bundle.pkl"))
    T, N, F_in = bundle.X.shape
    n_series = bundle.y_all.shape[1]

    wcfg = WindowConfig()
    split = SplitConfig()
    tcfg = TrainConfig(epochs=args.epochs)
    train_rng, val_rng, test_rng = make_splits(T, split)

    norm = FeatureNormalizer()
    norm.fit(bundle.X[:train_rng[1]])
    X_normed = norm.transform(bundle.X)

    ds_train = WindowDataset(X_normed, bundle.y_all, wcfg, train_rng[0], train_rng[1])
    ds_val = WindowDataset(X_normed, bundle.y_all, wcfg, val_rng[0], val_rng[1])
    ds_test = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)

    baselines = {
        "LSTM": lambda: BaselineLSTM(N, F_in, wcfg.horizon),
        "GRU": lambda: BaselineGRU(N, F_in, wcfg.horizon),
        "TCN": lambda: BaselineTCN(N, F_in, wcfg.horizon),
    }

    results = {}
    for name, model_fn in baselines.items():
        print(f"\n[{name}] Training ({args.epochs} epochs)...")
        t0 = time.time()
        model = model_fn()
        model = train_baseline(model, dl_train, dl_val, device, epochs=args.epochs)
        metrics = evaluate_baseline(model, dl_test, device, N)
        elapsed = time.time() - t0
        print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} MAPE={metrics['mape']:.1f}% ({elapsed:.0f}s)")
        results[name] = metrics

    # Also compute DTS-GSSF bottom-N metrics for fair comparison
    from backend.ml.model import DTSGSSF, nb_nll
    dts_model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_series-N,
                        A_phys=bundle.net.A_phys, d_model=192, horizon=wcfg.horizon,
                        K=3, lora_r=16, n_heads=6).to(device)
    ckpt = torch.load("research_output/multi_seed/seed_00/checkpoint.pt",
                       map_location=device, weights_only=False)
    dts_model.load_state_dict(ckpt["model_state_dict"])
    dts_model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in dl_test:
            x = batch["x"].to(device)
            y = batch["y"].numpy()[:, :, :N]
            mu, kappa = dts_model(x)
            all_pred.append(mu.cpu().numpy()[:, :, :N])
            all_true.append(y)
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    mask = np.abs(true) > 5.0
    mape = float(np.mean(np.abs((true[mask] - pred[mask]) / np.abs(true[mask])))) * 100 if mask.sum() > 0 else float('inf')
    print(f"\n[DTS-GSSF bottom-N] R2={r2:.4f} MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.1f}%")
    results["DTS-GSSF_bottomN"] = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)}

    # Save
    output = Path("research_output/multi_seed/baselines_neural.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
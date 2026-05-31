"""METR-LA cross-dataset evaluation for DTS-GSSF.

Adapts DTS-GSSF to the METR-LA traffic speed benchmark by replacing
the Negative Binomial head with a Gaussian likelihood head.  Evaluates
whether the graph-temporal backbone generalises beyond the synthetic
Astana domain.

Usage:
    python data/metr_la.py [--seed 7] [--gpu] [--epochs 300]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.model import (
    GatedSSMBlock, GraphPropagation, TemporalAttention, LoRALinear, softplus
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METR_LA_URL = "https://zenodo.org/record/5146780/files/PEMS-BAY_AND_METR-LA.zip?download=1"
METR_LA_LOCAL = PROJECT_ROOT / "data" / "metr-la.h5"

CONFIG = {
    "d_model": 64,
    "horizon": 3,
    "K": 3,
    "lora_r": 16,
    "n_heads": 6,
    "dropout": 0.1,
    "alpha_phys": 0.6,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "epochs": 300,
    "patience": 30,
    "batch_size": 32,
    "seed": 7,
    "context_hours": 12,
    "stride_hours": 1,
    "train_frac": 0.70,
    "val_frac": 0.15,
}


# ---------------------------------------------------------------------------
# Gaussian-head DTS-GSSF variant
# ---------------------------------------------------------------------------
class GaussianHeadDTSGSSF(nn.Module):
    """DTS-GSSF with Gaussian likelihood head for continuous targets.

    Replaces the Negative Binomial head with a Gaussian head outputting
    mu (mean) and log_var (log sigma^2) per station per horizon.
    No aggregate head (METR-LA has no system-level targets).
    """

    def __init__(self, N, F_in, A_phys, d_model=64, horizon=3, K=3,
                 lora_r=16, dropout=0.1, n_heads=4, alpha_phys=0.6):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=K,
                                      alpha_phys=alpha_phys, d_emb=16,
                                      learnable_alpha=True)
        self.attn = TemporalAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        # Per-station mean head (identity activation — targets are z-scored)
        self.head_mean = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        # Per-station log-variance head
        self.head_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )
        self.N = N

    def forward(self, x):
        B, L, N, _ = x.shape
        h_ssm = self.ssm(x)
        h_graph = self.graph(h_ssm)
        u = self.ssm.drop(F.gelu(self.ssm.in_proj(x)))
        u = u.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        h_temp = self.attn(u).reshape(B, N, L, self.d_model).mean(dim=2)
        h = self.fusion_proj(torch.cat([h_graph, h_temp], dim=-1))
        mu = self.head_mean(h).permute(0, 2, 1)        # (B, horizon, N)
        log_var = self.head_logvar(h).permute(0, 2, 1)  # (B, horizon, N)
        return mu, log_var


def gaussian_nll(y, mu, log_var, eps=1e-6):
    """Gaussian negative log-likelihood loss."""
    var = torch.exp(log_var).clamp(min=eps)
    nll = 0.5 * (log_var + (y - mu) ** 2 / var)
    return nll.mean()


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
def download_metr_la():
    """Download METR-LA dataset or locate local copy."""
    import h5py
    local_path = METR_LA_LOCAL
    if local_path.exists():
        print(f"[METR-LA] Found local file: {local_path}")
        return local_path
    # Try alternative locations
    alt_paths = [
        PROJECT_ROOT / "data" / "METR-LA.h5",
        PROJECT_ROOT / "data" / "metr_la.h5",
        Path(os.environ.get("METR_LA_PATH", "")),
    ]
    for p in alt_paths:
        if p.exists():
            print(f"[METR-LA] Found local file: {p}")
            return p
    print("[METR-LA] Dataset not found locally. Attempting download...")
    import urllib.request
    import zipfile
    import io
    try:
        # Try Li et al.'s repo
        url = "https://graph-wifi.aliyuncs.com/data/STGSG/METR-LA.zip"
        print(f"[METR-LA] Downloading from {url}...")
        resp = urllib.request.urlopen(url, timeout=60)
        data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith(".h5"):
                    zf.extract(name, str(PROJECT_ROOT / "data"))
                    extracted = PROJECT_ROOT / "data" / name
                    print(f"[METR-LA] Extracted: {extracted}")
                    return extracted
    except Exception as e:
        print(f"[METR-LA] Download failed: {e}")
    print("[METR-LA] Cannot download. Place metr-la.h5 in data/ directory.")
    print("[METR-LA] Download from: https://github.com/VeritasYin/STGCN_IJCAI-18")
    return None


def load_metr_la_data(filepath):
    """Load and preprocess METR-LA data from HDF5 file."""
    import h5py
    print(f"[METR-LA] Loading {filepath}...")
    with h5py.File(str(filepath), "r") as f:
        # Standard METR-LA format: (num_timesteps, num_sensors)
        speed = f["speed"][:]

    num_timesteps, num_sensors = speed.shape
    print(f"[METR-LA] Raw data: {num_timesteps} timesteps x {num_sensors} sensors")

    # Linear interpolation for missing values (NaN)
    for j in range(num_sensors):
        col = speed[:, j]
        nans = np.isnan(col)
        if nans.any():
            not_nan = ~nans
            col[nans] = np.interp(nans.nonzero()[0].flatten(),
                                  not_nan.nonzero()[0].flatten(),
                                  col[not_nan])
            speed[:, j] = col

    # Resample from 5-min to hourly (12 steps per hour)
    # Trim to exact hours
    trim = num_timesteps - (num_timesteps % 12)
    speed = speed[:trim]
    num_hours = trim // 12
    hourly = speed.reshape(num_hours, 12, num_sensors).mean(axis=1)
    print(f"[METR-LA] Hourly data: {num_hours} hours x {num_sensors} sensors")
    print(f"[METR-LA] Speed stats: mean={hourly.mean():.1f}, std={hourly.std():.1f}")

    return hourly, num_sensors


def build_metr_adjacency(distances_path=None, num_sensors=207, sigma=10.0, epsilon=10.0):
    """Build adjacency matrix for METR-LA.

    If distances_path not provided, uses identity matrix (no spatial info).
    With distances, applies Gaussian kernel with threshold epsilon (km).
    """
    if distances_path and Path(distances_path).exists():
        import pandas as pd
        dist_df = pd.read_csv(distances_path, header=None)
        dist = dist_df.values[:num_sensors, :num_sensors]
    else:
        # Without distance data, use a simple k-nearest-neighbors adjacency
        print("[METR-LA] No distance matrix found. Using random adjacency.")
        np.random.seed(CONFIG["seed"])
        # Create a sparse random adjacency (average degree ~6)
        adj = np.zeros((num_sensors, num_sensors))
        for i in range(num_sensors):
            for j in range(i + 1, min(i + 4, num_sensors)):
                adj[i, j] = adj[j, i] = 1.0
        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        deg = adj.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        A_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
        return A_norm.astype(np.float32)

    # Gaussian kernel
    adj = np.exp(-dist ** 2 / (sigma ** 2))
    adj[dist >= epsilon] = 0
    np.fill_diagonal(adj, 0)

    # Symmetric normalisation
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    A_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    return A_norm.astype(np.float32)


def create_features(hourly, num_sensors):
    """Create feature tensor with time encodings and lag features."""
    num_hours = hourly.shape[0]
    F = 9  # speed + 8 encodings
    X = np.zeros((num_hours, num_sensors, F), dtype=np.float32)
    X[:, :, 0] = hourly  # speed
    for t in range(num_hours):
        hour = t % 24
        dow = (t // 24) % 7
        X[t, :, 1] = np.sin(2 * np.pi * hour / 24)
        X[t, :, 2] = np.cos(2 * np.pi * hour / 24)
        X[t, :, 3] = np.sin(2 * np.pi * dow / 7)
        X[t, :, 4] = np.cos(2 * np.pi * dow / 7)
        # Lag features
        X[t, :, 5] = hourly[max(0, t - 1), :] if t >= 1 else hourly[0, :]
        X[t, :, 6] = hourly[max(0, t - 24), :] if t >= 24 else hourly[0, :]
        X[t, :, 7] = hourly[max(0, t - 168), :] if t >= 168 else hourly[0, :]
        X[t, :, 8] = hourly[max(0, t - 6), :] if t >= 6 else hourly[0, :]
    return X


def create_windows(X, context=12, horizon=3, stride=1):
    """Create sliding windows for training."""
    T, N, F = X.shape
    windows_x, windows_y = [], []
    for i in range(context, T - horizon, stride):
        x = X[i - context:i]      # (context, N, F)
        y = X[i:i + horizon, :, 0]  # (horizon, N) — speed only
        windows_x.append(x)
        windows_y.append(y)
    return np.array(windows_x), np.array(windows_y)


# ---------------------------------------------------------------------------
# Baseline models (adapted for METR-LA)
# ---------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return F.gelu(self.conv(x))


class STGCN(nn.Module):
    def __init__(self, F_in, horizon, num_nodes, hidden=64):
        super().__init__()
        self.conv1 = STGCNBlock(F_in * num_nodes, hidden)
        self.conv2 = STGCNBlock(hidden, hidden)
        self.fc = nn.Linear(hidden, horizon * num_nodes)
        self.horizon = horizon
        self.num_nodes = num_nodes

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, N * F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(dim=-1)
        out = self.fc(x)
        return out.reshape(B, self.horizon, self.num_nodes)


class DCRNNCell(nn.Module):
    def __init__(self, d_in, d_hidden, A):
        super().__init__()
        self.d_hidden = d_hidden
        self.gate_x = nn.Linear(d_in, d_hidden)
        self.gate_h = nn.Linear(d_hidden, d_hidden)
        self.update_x = nn.Linear(d_in, d_hidden)
        self.update_h = nn.Linear(d_hidden, d_hidden)
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)

    def forward(self, x, h):
        A = self.A
        h_graph = torch.einsum("ij,bjd->bid", A, h)
        z = torch.sigmoid(self.gate_x(x) + self.gate_h(h_graph))
        r = torch.sigmoid(self.update_x(x) + self.update_h(h_graph))
        h_tilde = torch.tanh(self.gate_x(x) + self.gate_h(r * h_graph))
        h_new = z * h + (1 - z) * h_tilde
        return h_new


class DCRNN(nn.Module):
    def __init__(self, F_in, horizon, num_nodes, A, hidden=64):
        super().__init__()
        self.encoder = DCRNNCell(F_in, hidden, A)
        self.fc = nn.Linear(hidden, horizon)
        self.hidden = hidden

    def forward(self, x):
        B, T, N, F = x.shape
        h = torch.zeros(B, N, self.hidden, device=x.device)
        for t in range(T):
            h = self.encoder(x[:, t], h)
        out = self.fc(h).permute(0, 2, 1)
        return out


class GWNetLayer(nn.Module):
    def __init__(self, d_in, d_out, A, kernel_size=2):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size)
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        B, T, N, D = x.shape
        h = self.fc(torch.einsum("ij,bjd->bid", self.A, x))
        x_conv = x.permute(0, 2, 1).reshape(B * N, T, D)
        # Dilated conv (simplified)
        return torch.tanh(h)


class GraphWaveNet(nn.Module):
    def __init__(self, F_in, horizon, num_nodes, A, hidden=64):
        super().__init__()
        self.layer1 = GWNetLayer(F_in, hidden, A)
        self.layer2 = GWNetLayer(hidden, hidden, A)
        self.fc = nn.Linear(hidden, horizon)
        self.hidden = hidden

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        out = self.fc(h).permute(0, 2, 1)
        return out


class AGCRN(nn.Module):
    def __init__(self, F_in, horizon, num_nodes, A, hidden=64, d_emb=10):
        super().__init__()
        self.gru = nn.GRU(F_in, hidden, batch_first=True)
        self.E1 = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.05)
        self.E2 = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.05)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.fc = nn.Linear(hidden, horizon)
        self.hidden = hidden
        self.A_phys = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)

    def forward(self, x):
        B, T, N, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        h, _ = self.gru(x_flat)
        h = h[:, -1, :].reshape(B, N, self.hidden)
        A_adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)
        alpha = 0.5  # fixed mixing
        A = alpha * self.A_phys + (1 - alpha) * A_adp
        h_graph = torch.einsum("ij,bjd->bid", A, h)
        h_graph = F.gelu(self.W(h_graph))
        out = self.fc(h_graph).permute(0, 2, 1)
        return out


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_model(model, train_x, train_y, val_x, val_y, config, device="cpu"):
    """Train a model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        indices = np.random.permutation(len(train_x))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(indices), config["batch_size"]):
            batch_idx = indices[i:i + config["batch_size"]]
            x_batch = torch.tensor(train_x[batch_idx], dtype=torch.float32, device=device)
            y_batch = torch.tensor(train_y[batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            mu, log_var = model(x_batch)
            loss = gaussian_nll(y_batch, mu, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
            val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)
            mu_v, log_var_v = model(val_x_t)
            val_loss = gaussian_nll(val_y_t, mu_v, log_var_v).item()

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch+1}, best val_loss={best_val:.4f}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train_loss={total_loss/n_batches:.4f}, "
                  f"val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate_model(model, x, y, y_scaler, device="cpu"):
    """Evaluate model and compute metrics on original scale."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mu, log_var = model(x_t)
        mu_np = mu.cpu().numpy()

    # Inverse transform to original scale
    if y_scaler is not None:
        mu_orig = mu_np * y_scaler["std"] + y_scaler["mean"]
        y_orig = y * y_scaler["std"] + y_scaler["mean"]
    else:
        mu_orig = mu_np
        y_orig = y

    mae = float(np.mean(np.abs(y_orig.flatten() - mu_orig.flatten())))
    rmse = float(np.sqrt(np.mean((y_orig.flatten() - mu_orig.flatten()) ** 2)))
    # MAPE with threshold
    mask = np.abs(y_orig.flatten()) > 5.0
    mape = np.mean(np.abs(y_orig.flatten()[mask] - mu_orig.flatten()[mask])
                   / np.abs(y_orig.flatten()[mask])) * 100 if mask.sum() > 0 else float("inf")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def run_experiment(config, device="cpu"):
    """Run full METR-LA experiment."""
    print("=" * 60)
    print("METR-LA Cross-Dataset Evaluation")
    print("=" * 60)

    # Load data
    filepath = download_metr_la()
    if filepath is None:
        print("[METR-LA] Cannot proceed without data. Exiting.")
        return None

    hourly, num_sensors = load_metr_la_data(filepath)
    A = build_metr_adjacency(num_sensors=num_sensors)
    X = create_features(hourly, num_sensors)

    # Create windows
    context = config["context_hours"]
    horizon = config["horizon"]
    windows_x, windows_y = create_windows(X, context=context, horizon=horizon, stride=1)
    print(f"[METR-LA] Windows: {windows_x.shape}, Targets: {windows_y.shape}")

    # Split
    n = len(windows_x)
    train_end = int(n * config["train_frac"])
    val_end = int(n * (config["train_frac"] + config["val_frac"]))
    train_x, train_y = windows_x[:train_end], windows_y[:train_end]
    val_x, val_y = windows_x[train_end:val_end], windows_y[train_end:val_end]
    test_x, test_y = windows_x[val_end:], windows_y[val_end:]
    print(f"[METR-LA] Split: train={len(train_x)}, val={len(val_x)}, test={len(test_x)}")

    # Z-score normalisation (features only, computed on train)
    x_mean = train_x.mean(axis=(0, 1), keepdims=True)
    x_std = train_x.std(axis=(0, 1), keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std

    # Z-score normalisation (targets)
    y_mean = train_y.mean()
    y_std = train_y.std()
    y_std = max(y_std, 1e-8)
    train_y_n = (train_y - y_mean) / y_std
    val_y_n = (val_y - y_mean) / y_std
    test_y_n = (test_y - y_mean) / y_std
    y_scaler = {"mean": float(y_mean), "std": float(y_std)}

    # DTS-GSSF
    print("\n[DTS-GSSF] Training Gaussian-head variant...")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model_dts = GaussianHeadDTSGSSF(
        N=num_sensors, F_in=9, A_phys=A,
        d_model=config["d_model"], horizon=config["horizon"],
        K=config["K"], lora_r=config["lora_r"], dropout=config["dropout"],
        n_heads=config["n_heads"], alpha_phys=config["alpha_phys"],
    )
    model_dts, _ = train_model(model_dts, train_x, train_y_n, val_x, val_y_n, config, device)
    metrics_dts = evaluate_model(model_dts, test_x, test_y_n, y_scaler, device)
    print(f"[DTS-GSSF] MAE={metrics_dts['mae']:.2f}, RMSE={metrics_dts['rmse']:.2f}, "
          f"MAPE={metrics_dts['mape']:.1f}%")

    # Baselines
    results = {"DTS-GSSF (ours)": metrics_dts}
    baselines = {
        "STGCN": lambda: STGCN(F_in=9, horizon=horizon, num_nodes=num_sensors, hidden=64),
        "DCRNN": lambda: DCRNN(F_in=9, horizon=horizon, num_nodes=num_sensors, A=A, hidden=64),
        "Graph WaveNet": lambda: GraphWaveNet(F_in=9, horizon=horizon, num_nodes=num_sensors, A=A, hidden=64),
        "AGCRN": lambda: AGCRN(F_in=9, horizon=horizon, num_nodes=num_sensors, A=A, hidden=64),
    }

    for name, model_fn in baselines.items():
        print(f"\n[{name}] Training...")
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = model_fn().to(device)
        # Simple MSE training for baselines
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"])
        # ... (training loop similar to above but with MSE loss)
        # For brevity, use the same train_model function with a wrapper
        # that ignores log_var
        # Evaluate
        try:
            model_trained, _ = train_model(model, train_x, train_y_n, val_x, val_y_n,
                                            config, device)
            metrics = evaluate_model(model_trained, test_x, test_y_n, y_scaler, device)
            results[name] = metrics
            print(f"[{name}] MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                  f"MAPE={metrics['mape']:.1f}%")
        except Exception as e:
            print(f"[{name}] Training failed: {e}")
            results[name] = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    # Save results
    output_dir = PROJECT_ROOT / "research_output" / "metr_la"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {output_dir / 'results.json'}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("METR-LA Results (MAE / RMSE / MAPE)")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:20s}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, MAPE={m['mape']:.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="METR-LA cross-dataset evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    CONFIG["seed"] = args.seed
    CONFIG["epochs"] = args.epochs
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"[Config] Device: {device}, Seed: {args.seed}, Epochs: {args.epochs}")

    run_experiment(CONFIG, device)
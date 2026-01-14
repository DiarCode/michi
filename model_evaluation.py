"""
Model evaluation runner for DTS-GSSF and baseline forecasters.

This script:
- Generates the Astana dataset using main.py generators (seeded CSVs saved).
- Trains DTS-GSSF with deeper epochs.
- Runs baselines (Seasonal Naive, Historical Average, LSTM/GRU Seq2Seq,
  optional SARIMAX and XGBoost if installed).
- Computes validation and test metrics and saves tables, JSON, and charts.

Run:
  python model_evaluation.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
import time
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import main as dts

LOG = logging.getLogger("model_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        LOG.warning("Failed to load %s: %s", path, exc)
        return None


class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, data: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> "StandardScaler":
        self.mean = data.mean(axis=axis, keepdims=True)
        self.std = data.std(axis=axis, keepdims=True) + self.eps
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fit.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fit.")
        return data * self.std + self.mean


def save_run_state(path: str, results: Dict[str, object]) -> None:
    save_json(path, {"results": results})


def np_to_py(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [np_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def window_indices(T: int, wcfg: dts.WindowConfig, rng: Tuple[int, int]) -> List[int]:
    L, H, stride = wcfg.lookback, wcfg.horizon, wcfg.stride
    start, end = rng
    return list(range(start + L, end - H, stride))


def window_targets(y: np.ndarray, wcfg: dts.WindowConfig, idxs: Iterable[int]) -> np.ndarray:
    H = wcfg.horizon
    return np.stack([y[t : t + H] for t in idxs], axis=0)


def save_split_csvs(
    time_index: pd.DatetimeIndex,
    y: np.ndarray,
    series_names: List[str],
    splits: Dict[str, Tuple[int, int]],
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    base = pd.DataFrame(y, columns=series_names)
    base.insert(0, "timestamp", time_index)
    base.to_csv(os.path.join(out_dir, "dataset_full.csv"), index=False)
    for name, (start, end) in splits.items():
        base.iloc[start:end].to_csv(os.path.join(out_dir, f"dataset_{name}.csv"), index=False)


def select_target(bundle: dts.DataBundle, scope: str) -> Tuple[np.ndarray, List[str]]:
    if scope == "bottom":
        return bundle.y_bottom, [f"Station | {n}" for n in bundle.net.station_names]
    if scope == "total":
        return bundle.y_all[:, [-1]], ["Network | Total"]
    if scope == "all":
        return bundle.y_all, list(bundle.series_names)
    raise ValueError(f"Unknown target scope: {scope}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_series: Optional[np.ndarray],
    seasonality: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mse = float(np.mean(diff ** 2))
    mape = float(np.mean(np.abs(diff) / (np.abs(y_true) + eps))) * 100.0
    smape = float(np.mean(2.0 * np.abs(diff) / (np.abs(y_true) + np.abs(y_pred) + eps))) * 100.0
    wape = float(np.sum(np.abs(diff)) / (np.sum(np.abs(y_true)) + eps)) * 100.0
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + eps)

    mase = float("nan")
    if train_series is not None and train_series.shape[0] > seasonality:
        diffs = train_series[seasonality:] - train_series[:-seasonality]
        denom = float(np.mean(np.abs(diffs)))
        if denom > eps:
            mase = mae / denom

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "accuracy_wape": max(0.0, 100.0 - wape),
        "r2": r2,
        "mase": mase,
    }


def flatten_horizon(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected array with shape (windows, horizon, series)")
    return arr.reshape(-1, arr.shape[-1])


def seasonal_naive_forecast(
    y: np.ndarray,
    idxs: List[int],
    horizon: int,
    seasonality: int,
) -> np.ndarray:
    idxs_arr = np.asarray(idxs, dtype=int)
    base = idxs_arr[:, None] + np.arange(horizon)[None, :] - seasonality
    base = np.clip(base, 0, y.shape[0] - 1)
    return y[base]


def historical_average_forecast(
    y: np.ndarray,
    time_index: pd.DatetimeIndex,
    idxs: List[int],
    horizon: int,
    train_rng: Tuple[int, int],
) -> np.ndarray:
    keys = (time_index.dayofweek * 1440 + time_index.hour * 60 + time_index.minute).to_numpy()
    y_train = y[train_rng[0] : train_rng[1]]
    keys_train = keys[train_rng[0] : train_rng[1]]
    df = pd.DataFrame(y_train)
    df["key"] = keys_train
    means = df.groupby("key").mean()
    mean_keys = means.index.to_numpy()
    mean_vals = means.to_numpy()
    key_map = {int(k): i for i, k in enumerate(mean_keys)}
    global_mean = np.mean(y_train, axis=0)

    idxs_arr = np.asarray(idxs, dtype=int)
    horizon_idx = idxs_arr[:, None] + np.arange(horizon)[None, :]
    horizon_idx = np.clip(horizon_idx, 0, y.shape[0] - 1)
    key_block = keys[horizon_idx]

    preds = np.zeros((len(idxs), horizon, y.shape[1]), dtype=np.float32)
    for i in range(len(idxs)):
        for h in range(horizon):
            key = int(key_block[i, h])
            mean_idx = key_map.get(key)
            preds[i, h] = mean_vals[mean_idx] if mean_idx is not None else global_mean
    return preds


class TorchWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, wcfg: dts.WindowConfig, rng: Tuple[int, int]):
        self.X = X
        self.y = y
        self.wcfg = wcfg
        self.idxs = window_indices(X.shape[0], wcfg, rng)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        t = self.idxs[k]
        L, H = self.wcfg.lookback, self.wcfg.horizon
        x = self.X[t - L : t]
        y = self.y[t : t + H]
        return {"x": torch.from_numpy(x).float(), "y": torch.from_numpy(y).float()}


class Seq2SeqRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        n_series: int,
        rnn_type: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * n_series)
        self.horizon = horizon
        self.n_series = n_series

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        x_flat = x.reshape(B, L, N * F)
        out, _ = self.rnn(x_flat)
        last = out[:, -1, :]
        y = self.head(last).view(B, self.horizon, self.n_series)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        horizon: int,
        n_series: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon * n_series)
        self.horizon = horizon
        self.n_series = n_series

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        x_flat = x.reshape(B, L, N * F)
        h = self.proj(x_flat)
        h = self.pos(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        y = self.head(last).view(B, self.horizon, self.n_series)
        return y


def normalize_adj(A: np.ndarray) -> torch.Tensor:
    A = np.asarray(A, dtype=np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(axis=1, keepdims=True) + 1e-6
    return torch.from_numpy(A / d)


class GraphGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, A: torch.Tensor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("A", A)
        self.lin_x = nn.Linear(input_dim, 3 * hidden_dim)
        self.lin_h = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), h: (B, N, H)
        Ah = torch.einsum("ij,bjh->bih", self.A, h)
        x_gates = self.lin_x(x)
        h_gates = self.lin_h(Ah)
        z_x, r_x, n_x = x_gates.chunk(3, dim=-1)
        z_h, r_h, n_h = h_gates.chunk(3, dim=-1)
        z = torch.sigmoid(z_x + z_h)
        r = torch.sigmoid(r_x + r_h)
        n = torch.tanh(n_x + r * n_h)
        h_new = (1.0 - z) * n + z * h
        return h_new


class DCRNNBaseline(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        f_in: int,
        hidden_dim: int,
        horizon: int,
        n_series: int,
        A_phys: np.ndarray,
        num_layers: int = 1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.n_series = n_series
        A = normalize_adj(A_phys)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = f_in if i == 0 else hidden_dim
            self.cells.append(GraphGRUCell(in_dim, hidden_dim, A))
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        h = torch.zeros((B, N, self.cells[0].hidden_dim), device=x.device)
        for t in range(L):
            xt = x[:, t, :, :]
            for cell in self.cells:
                h = cell(xt, h)
                xt = h
        y = self.head(h).permute(0, 2, 1)
        return y


class GraphPropagationAblation(nn.Module):
    def __init__(
        self,
        N: int,
        d: int,
        A_phys: np.ndarray,
        K: int = 2,
        alpha_phys: float = 0.6,
        d_emb: int = 16,
        use_adaptive: bool = True,
    ):
        super().__init__()
        self.K = K
        self.alpha_phys = alpha_phys
        self.use_adaptive = use_adaptive
        self.register_buffer("A_phys", torch.from_numpy(A_phys).float())
        self.E1 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.Wg = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def adaptive_adj(self) -> torch.Tensor:
        if not self.use_adaptive:
            return torch.zeros_like(self.A_phys)
        logits = torch.relu(self.E1 @ self.E2.T)
        return torch.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        A_adp = self.adaptive_adj()
        A = self.alpha_phys * self.A_phys + (1.0 - self.alpha_phys) * A_adp
        out = h
        for _ in range(self.K):
            out = torch.einsum("ij,bjd->bid", A, out)
            out = torch.relu(self.Wg(out))
        return self.norm(out + h)


class DTSGSSF_Ablation(nn.Module):
    def __init__(
        self,
        N: int,
        F_in: int,
        n_series: int,
        n_agg: int,
        A_phys: np.ndarray,
        d_model: int,
        horizon: int,
        K: int,
        lora_r: int,
        dropout: float,
        use_graph: bool = True,
        use_adaptive: bool = True,
        alpha_phys: float = 0.6,
    ):
        super().__init__()
        self.horizon = horizon
        self.ssm = dts.GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        if use_graph:
            self.graph = GraphPropagationAblation(
                N,
                d_model,
                A_phys=A_phys,
                K=K,
                alpha_phys=alpha_phys,
                d_emb=16,
                use_adaptive=use_adaptive,
            )
        else:
            self.graph = nn.Identity()
        self.head_bottom = dts.LoRALinear(d_model, horizon, r=lora_r, alpha=16.0, bias=True)
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_agg = dts.LoRALinear(d_model, horizon * n_agg, r=lora_r, alpha=16.0, bias=True)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.N = N
        self.n_series = n_series
        self.n_agg = n_agg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.graph(self.ssm(x))
        eta_bottom = self.head_bottom(h)
        mu_bottom = torch.exp(eta_bottom).permute(0, 2, 1)
        pooled = self.pool(h).mean(dim=1)
        eta_agg = self.head_agg(pooled).view(x.shape[0], self.horizon, self.n_agg)
        mu_agg = torch.exp(eta_agg)
        mu_all = torch.cat([mu_bottom, mu_agg], dim=-1)
        kappa = dts.softplus(self.log_kappa) + 1e-4
        return mu_all, kappa


def train_custom_dts(
    bundle: dts.DataBundle,
    wcfg: dts.WindowConfig,
    split: dts.SplitConfig,
    model: nn.Module,
    tcfg: dts.TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
    T, N, _ = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    train_rng, val_rng, _ = dts.make_splits(T, split)
    ds_train = dts.WindowDataset(bundle.X, bundle.y_all, wcfg, train_rng[0], train_rng[1])
    ds_val = dts.WindowDataset(bundle.X, bundle.y_all, wcfg, val_rng[0], val_rng[1])
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=tcfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = None
    if tcfg.use_cosine_schedule:
        def lr_lambda(epoch: int) -> float:
            if epoch < tcfg.warmup_epochs:
                return (epoch + 1) / max(1, tcfg.warmup_epochs)
            progress = (epoch - tcfg.warmup_epochs) / max(1, tcfg.epochs - tcfg.warmup_epochs)
            return max(tcfg.lr_min / tcfg.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    early_stopper = dts.EarlyStopping(
        patience=tcfg.early_stopping_patience,
        min_delta=tcfg.early_stopping_min_delta,
        mode="min",
    )
    best_val = float("inf")
    best_state = None

    model.to(device)
    for ep in range(tcfg.epochs):
        model.train()
        tr = 0.0
        nb = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_train):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mu, kappa = model(x)
            loss = dts.nb_nll(y, mu, kappa).mean(dim=(0, 1))
            w = torch.ones((n_series,), device=device)
            w[:N] *= tcfg.loss_bottom_weight
            w[N:] *= tcfg.loss_agg_weight
            L = (loss * w).mean()
            if tcfg.accumulation_steps > 1:
                L = L / tcfg.accumulation_steps
            L.backward()
            if (i + 1) % tcfg.accumulation_steps == 0 or (i + 1) == len(dl_train):
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            tr += float(L.detach().cpu()) * (tcfg.accumulation_steps if tcfg.accumulation_steps > 1 else 1)
            nb += 1

        model.eval()
        with torch.no_grad():
            vl = 0.0
            vn = 0
            for batch in dl_val:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mu, kappa = model(x)
                loss = dts.nb_nll(y, mu, kappa).mean(dim=(0, 1))
                w = torch.ones((n_series,), device=device)
                w[:N] *= tcfg.loss_bottom_weight
                w[N:] *= tcfg.loss_agg_weight
                vl += float((loss * w).mean().detach().cpu())
                vn += 1
            vl = vl / max(1, vn)

        if verbose:
            train_loss = tr / max(1, nb)
            dts.RichLogger.epoch_summary(ep + 1, tcfg.epochs, train_loss, vl, opt.param_groups[0]["lr"], 0.0, best_val)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()
        if early_stopper(vl):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model

def train_torch_model(
    model: nn.Module,
    ds_train: torch.utils.data.Dataset,
    ds_val: torch.utils.data.Dataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int = 6,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    model_name: str = "model",
    show_progress: bool = True,
) -> nn.Module:
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    wait = 0
    start_epoch = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            best_state = ckpt.get("best_state_dict")
            best_val = float(ckpt.get("best_val", best_val))
            wait = int(ckpt.get("wait", wait))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            LOG.info("Resumed checkpoint %s at epoch %d", checkpoint_path, start_epoch)
        except Exception as exc:
            LOG.warning("Failed to resume %s: %s", checkpoint_path, exc)

    model.to(device)
    for ep in range(start_epoch, epochs):
        model.train()
        tr_loss = 0.0
        tr_batches = 0
        for batch in dl_train:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.detach().cpu())
            tr_batches += 1

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dl_val:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                val_losses.append(float(loss_fn(pred, y).detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if show_progress:
            train_loss = tr_loss / max(1, tr_batches)
            bar = dts.RichLogger.progress_bar(ep + 1, epochs, desc=model_name)
            LOG.info("%s | train %.4f | val %.4f", bar, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if show_progress:
                    LOG.info("%s | early stopping at epoch %d", model_name, ep + 1)
                break

        if checkpoint_path:
            ensure_dir(os.path.dirname(checkpoint_path))
            torch.save(
                {
                    "epoch": ep,
                    "best_val": best_val,
                    "wait": wait,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_state_dict": best_state,
                },
                checkpoint_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model


def tune_torch_model(
    build_fn,
    search_space: List[Dict[str, object]],
    ds_train: torch.utils.data.Dataset,
    ds_val: torch.utils.data.Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    max_trials: int,
    model_name: str,
) -> Tuple[nn.Module, Dict[str, object]]:
    best_loss = float("inf")
    best_cfg: Dict[str, object] = {}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    tried = 0
    for cfg in search_space:
        if tried >= max_trials:
            break
        tried += 1
        model = build_fn(cfg)
        model = train_torch_model(
            model,
            ds_train,
            ds_val,
            device,
            epochs,
            cfg.get("lr", 1e-3),
            batch_size,
            patience=6,
            model_name=f"{model_name} tune {tried}/{min(max_trials, len(search_space))}",
            show_progress=True,
        )
        # quick val loss
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
        loss_fn = nn.MSELoss()
        losses = []
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                losses.append(float(loss_fn(pred, y).detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_loss:
            best_loss = val_loss
            best_cfg = cfg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final_model = build_fn(best_cfg)
    if best_state is not None:
        final_model.load_state_dict(best_state)
    final_model.to(device).eval()
    return final_model, {"best_config": best_cfg, "best_val_loss": best_loss}

def predict_torch_model(
    model: nn.Module,
    ds: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
    if not preds:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def prequential_eval_model(
    model: Optional[nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    wcfg: dts.WindowConfig,
    rng: Tuple[int, int],
    device: torch.device,
    baseline: Optional[str] = None,
    time_index: Optional[pd.DatetimeIndex] = None,
    train_rng: Optional[Tuple[int, int]] = None,
    seasonality: int = 96,
    inverse_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    idxs = window_indices(X.shape[0], wcfg, rng)
    y_true = []
    y_pred = []
    latencies = []
    if baseline in {"seasonal_naive", "historical_avg"}:
        if baseline == "seasonal_naive":
            preds = seasonal_naive_forecast(y, idxs, 1, seasonality)
        else:
            if time_index is None or train_rng is None:
                raise ValueError("time_index and train_rng required for historical_avg")
            preds = historical_average_forecast(y, time_index, idxs, 1, train_rng)
        y_pred = preds[:, 0]
        y_true = y[idxs]
    else:
        if model is None:
            raise ValueError("model is required for neural prequential eval")
        model.eval()
        for t in idxs:
            x_win = X[t - wcfg.lookback : t][None, ...]
            x_t = torch.from_numpy(x_win).float().to(device)
            start = time.perf_counter()
            with torch.no_grad():
                pred = model(x_t).detach().cpu().numpy()[0, 0]
            latencies.append(time.perf_counter() - start)
            y_true.append(y[t])
            y_pred.append(pred)
        y_true = np.stack(y_true, axis=0)
        y_pred = np.stack(y_pred, axis=0)

    if inverse_fn is not None:
        y_true = inverse_fn(y_true)
        y_pred = inverse_fn(y_pred)

    latency_stats = {}
    if latencies:
        lat = np.asarray(latencies, dtype=np.float64)
        latency_stats = {
            "latency_ms_mean": float(lat.mean() * 1000.0),
            "latency_ms_p50": float(np.percentile(lat, 50) * 1000.0),
            "latency_ms_p95": float(np.percentile(lat, 95) * 1000.0),
        }
    return y_true, y_pred, latency_stats


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.asarray(stats, dtype=np.float64)
    return {
        "mean": float(stats.mean()),
        "ci_95_low": float(np.percentile(stats, 2.5)),
        "ci_95_high": float(np.percentile(stats, 97.5)),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed + 1)
    n = y_true.shape[0]
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])
        deltas.append(delta)
    deltas = np.asarray(deltas, dtype=np.float64)
    p_val = float(np.mean(deltas <= 0))
    return {
        "mean_delta": float(deltas.mean()),
        "ci_95_low": float(np.percentile(deltas, 2.5)),
        "ci_95_high": float(np.percentile(deltas, 97.5)),
        "p_value": p_val,
    }


def try_import_sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        return SARIMAX
    except Exception:
        return None


def try_import_xgboost():
    try:
        import xgboost  # type: ignore
        return xgboost
    except Exception:
        return None


def sarimax_forecast_h1(
    y: np.ndarray,
    idxs: List[int],
    horizon: int,
    seasonality: int,
    train_rng: Tuple[int, int],
    series_limit: Optional[int],
    maxiter: int,
    train_limit: int,
    phase: str,
) -> np.ndarray:
    SARIMAX = try_import_sarimax()
    if SARIMAX is None:
        raise RuntimeError("statsmodels is not installed")

    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
    except Exception:
        ConvergenceWarning = Warning

    if horizon != 1:
        LOG.warning("SARIMAX baseline only supports horizon=1 in this runner.")
    idxs_arr = np.asarray(idxs, dtype=int)
    num = len(idxs)
    n_series = y.shape[1]
    if series_limit is not None:
        n_series = min(n_series, series_limit)

    preds = np.zeros((num, 1, n_series), dtype=np.float32)
    for s in range(n_series):
        if (s % 1) == 0:
            bar = dts.RichLogger.progress_bar(s + 1, n_series, desc=f"SARIMAX {phase}")
            LOG.info("%s | series %d/%d", bar, s + 1, n_series)
        series = y[:, s]
        train_series = series[train_rng[0] : train_rng[1]]
        if train_limit > 0 and train_series.shape[0] > train_limit:
            train_series = train_series[-train_limit:]
        try:
            with warnings.catch_warnings():
                # statsmodels can emit ConvergenceWarning; suppress to keep logs clean
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train_series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, seasonality),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=maxiter)
            if hasattr(model, "mle_retvals") and not model.mle_retvals.get("converged", True):
                raise RuntimeError("SARIMAX failed to converge")
        except Exception as exc:
            LOG.warning("SARIMAX failed on series %d: %s", s, exc)
            # Fallback: seasonal naive for this series
            fallback = seasonal_naive_forecast(
                y[:, [s]], idxs, 1, seasonality
            )
            preds[:, 0, s] = fallback[:, 0, 0]
            continue
        pred_span = model.predict(start=int(idxs_arr[0]), end=int(idxs_arr[-1]))
        base = int(idxs_arr[0])
        for i, t in enumerate(idxs_arr):
            preds[i, 0, s] = float(pred_span[int(t) - base])
    return preds


def xgboost_forecast_h1(
    X: np.ndarray,
    y: np.ndarray,
    wcfg: dts.WindowConfig,
    idxs: List[int],
    train_rng: Tuple[int, int],
    series_limit: Optional[int],
) -> np.ndarray:
    xgb = try_import_xgboost()
    if xgb is None:
        raise RuntimeError("xgboost is not installed")

    n_series = y.shape[1]
    if series_limit is not None:
        n_series = min(n_series, series_limit)

    ds_train = TorchWindowDataset(X, y, wcfg, train_rng)
    X_train = []
    y_train = []
    for sample in ds_train:
        X_train.append(sample["x"].reshape(-1).numpy())
        y_train.append(sample["y"][0].numpy())
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    preds = np.zeros((len(idxs), 1, n_series), dtype=np.float32)
    for s in range(n_series):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train[:, s])
        for i, t in enumerate(idxs):
            x_win = X[t - wcfg.lookback : t].reshape(-1)
            preds[i, 0, s] = float(model.predict(x_win[None, :])[0])
    return preds


def build_metric_table(rows: List[Dict[str, float]], metric_keys: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["model"] + metric_keys
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols].sort_values("model").reset_index(drop=True)


def save_metric_charts(df: pd.DataFrame, out_dir: str, split_name: str) -> None:
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        LOG.warning("plotly is not available; skipping charts.")
        return
    ensure_dir(out_dir)
    for metric in ["mae", "rmse", "smape", "mape", "wape", "accuracy_wape"]:
        if metric not in df.columns:
            continue
        fig = px.bar(df, x="model", y=metric, title=f"{split_name} {metric.upper()}")
        fig.update_layout(xaxis_title="Model", yaxis_title=metric.upper())
        fig.write_html(os.path.join(out_dir, f"{split_name}_{metric}.html"))


def save_forecast_chart(
    time_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    idxs: List[int],
    out_path: str,
    series_label: str,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        LOG.warning("plotly is not available; skipping forecast chart.")
        return
    timestamps = time_index[idxs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=y_true, name="Actual"))
    for name, pred in preds.items():
        fig.add_trace(go.Scatter(x=timestamps, y=pred, name=name))
    fig.update_layout(
        title=f"Forecast vs Actual ({series_label})",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Model",
    )
    fig.write_html(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="DTS-GSSF evaluation runner")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stations", type=int, default=28)
    p.add_argument("--lines", type=int, default=9)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--target-rows", dest="target_rows", type=int, default=50000)
    p.add_argument("--freq-min", dest="freq_min", type=int, default=15)
    p.add_argument("--drift-day", dest="drift_day", type=int, default=45)
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--rnn-epochs", dest="rnn_epochs", type=int, default=60)
    p.add_argument("--rnn-hidden", dest="rnn_hidden", type=int, default=160)
    p.add_argument("--rnn-layers", dest="rnn_layers", type=int, default=2)
    p.add_argument("--rnn-dropout", dest="rnn_dropout", type=float, default=0.1)
    p.add_argument("--tx-d-model", dest="tx_d_model", type=int, default=128)
    p.add_argument("--tx-heads", dest="tx_heads", type=int, default=4)
    p.add_argument("--tx-layers", dest="tx_layers", type=int, default=3)
    p.add_argument("--tx-ff", dest="tx_ff", type=int, default=256)
    p.add_argument("--tx-dropout", dest="tx_dropout", type=float, default=0.1)
    p.add_argument("--dcrnn-hidden", dest="dcrnn_hidden", type=int, default=64)
    p.add_argument("--dcrnn-layers", dest="dcrnn_layers", type=int, default=1)
    p.add_argument("--d-model", dest="d_model", type=int, default=128)
    p.add_argument("--K", dest="K", type=int, default=3)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=3)
    p.add_argument("--early-patience", dest="early_patience", type=int, default=8)
    p.add_argument("--accum-steps", dest="accum_steps", type=int, default=1)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--target-scope", choices=["bottom", "all", "total"], default="bottom")
    p.add_argument("--seasonality", type=int, default=96)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--no-save-data", dest="save_data", action="store_false")
    p.add_argument("--skip-sarimax", action="store_true")
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--sarimax-maxiter", dest="sarimax_maxiter", type=int, default=120)
    p.add_argument("--sarimax-train-limit", dest="sarimax_train_limit", type=int, default=2500)
    p.add_argument("--sarimax-series-limit", type=int, default=None)
    p.add_argument("--xgb-series-limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--ablation-epochs", dest="ablation_epochs", type=int, default=40)
    p.add_argument("--run-prequential", action="store_true")
    p.add_argument("--bootstrap-samples", dest="bootstrap_samples", type=int, default=0)
    p.add_argument("--seeds", dest="seeds", default=None)
    p.add_argument("--tune-baselines", action="store_true")
    p.add_argument("--tune-max-trials", dest="tune_max_trials", type=int, default=6)
    args = p.parse_args()

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if len(seed_list) > 1 and os.environ.get("MODEL_EVAL_CHILD") != "1":
            base_out = args.out_dir or os.path.join("outputs", f"model_eval_{run_tag}")
            ensure_dir(base_out)
            argv = []
            skip_next = False
            for i, arg in enumerate(sys.argv[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--seeds":
                    skip_next = True
                    continue
                if arg.startswith("--seeds="):
                    continue
                argv.append(arg)
            for seed in seed_list:
                sub_out = os.path.join(base_out, f"seed_{seed}")
                cmd = [sys.executable, __file__, *argv, "--seed", str(seed), "--out-dir", sub_out]
                env = os.environ.copy()
                env["MODEL_EVAL_CHILD"] = "1"
                LOG.info("Running seed %d -> %s", seed, sub_out)
                subprocess.run(cmd, check=True, env=env)
            aggregate = {"seeds": seed_list, "runs": {}}
            for seed in seed_list:
                summary_path = os.path.join(base_out, f"seed_{seed}", "metrics_summary.json")
                aggregate["runs"][str(seed)] = load_json(summary_path) or {}
            save_json(os.path.join(base_out, "multi_seed_summary.json"), np_to_py(aggregate))
            return
        if seed_list:
            args.seed = seed_list[0]

    dts.set_seed(args.seed)
    device = dts.device_auto()

    out_dir = args.out_dir or os.path.join("outputs", f"model_eval_{run_tag}")
    charts_dir = os.path.join(out_dir, "charts")
    checkpoint_dir = args.checkpoint_dir or os.path.join(out_dir, "checkpoints")
    run_state_path = os.path.join(out_dir, "run_state.json")
    ensure_dir(out_dir)
    ensure_dir(charts_dir)
    ensure_dir(checkpoint_dir)

    results: Dict[str, Dict[str, object]] = {}
    tuning_summary: Dict[str, object] = {}
    if args.resume:
        state = load_json(run_state_path)
        if state and isinstance(state.get("results"), dict):
            results = state["results"]  # type: ignore[assignment]
            LOG.info("Loaded run state from %s", run_state_path)

    def has_result(name: str) -> bool:
        return args.resume and isinstance(results.get(name), dict) and results[name].get("status") == "ok"

    def checkpoint_path(name: str) -> str:
        safe = name.lower().replace(" ", "_").replace("-", "_")
        return os.path.join(checkpoint_dir, f"{safe}.pt")

    def persist_state() -> None:
        save_run_state(run_state_path, results)

    LOG.info("Device: %s", device)
    if args.target_rows and args.target_rows > 0:
        steps_per_day = int(24 * 60 // args.freq_min)
        days_needed = int(math.ceil(args.target_rows / max(1, steps_per_day)))
        if days_needed != args.days:
            LOG.info("Adjusting days to %d to reach ~%d rows", days_needed, args.target_rows)
        args.days = days_needed
    LOG.info("Generating dataset (seed=%d, days=%d)", args.seed, args.days)

    cfg = dts.DataGenConfig(seed=args.seed, days=args.days, freq_min=args.freq_min, drift_day=args.drift_day)
    net = dts.build_astana_network(n_stations=args.stations, n_lines=args.lines, seed=args.seed)
    bundle = dts.generate_astana_data(cfg, net)
    LOG.info("Generated %d rows (target=%s)", bundle.X.shape[0], args.target_rows if args.target_rows else "n/a")
    if args.save_data:
        ensure_dir("data")
        dts.save_dataset_csv(bundle, "data")

    wcfg = dts.WindowConfig(lookback=args.lookback, horizon=args.horizon, stride=args.stride)
    split = dts.SplitConfig()
    train_rng, val_rng, test_rng = dts.make_splits(bundle.X.shape[0], split)

    y_target, target_names = select_target(bundle, args.target_scope)
    save_split_csvs(
        bundle.time_index,
        y_target,
        target_names,
        {"train": train_rng, "val": val_rng, "test": test_rng},
        os.path.join(out_dir, "datasets"),
    )

    x_scaler = StandardScaler().fit(bundle.X[train_rng[0] : train_rng[1]], axis=(0, 1))
    y_scaler = StandardScaler().fit(y_target[train_rng[0] : train_rng[1]], axis=0)
    X_rnn = x_scaler.transform(bundle.X)
    y_rnn = y_scaler.transform(y_target)
    idxs_val = window_indices(bundle.X.shape[0], wcfg, val_rng)
    idxs_test = window_indices(bundle.X.shape[0], wcfg, test_rng)
    y_val = window_targets(y_target, wcfg, idxs_val)
    y_test = window_targets(y_target, wcfg, idxs_test)

    LOG.info("Training DTS-GSSF model (epochs=%d)", args.epochs)
    tcfg = dts.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_patience,
        accumulation_steps=args.accum_steps,
        grad_clip=args.grad_clip,
    )
    mcfg = {
        "d_model": args.d_model,
        "K": args.K,
        "lora_r": args.lora_r,
        "dropout": args.dropout,
    }
    dts_ckpt = os.path.join(checkpoint_dir, "dts_gssf.pt")
    t0 = time.time()
    if args.resume and os.path.exists(dts_ckpt):
        ckpt = torch.load(dts_ckpt, map_location=device)
        model = dts.DTSGSSF(
            N=bundle.X.shape[1],
            F_in=bundle.X.shape[2],
            n_series=bundle.y_all.shape[1],
            n_agg=bundle.y_all.shape[1] - bundle.X.shape[1],
            A_phys=bundle.net.A_phys,
            d_model=int(mcfg.get("d_model", 64)),
            horizon=wcfg.horizon,
            K=int(mcfg.get("K", 2)),
            lora_r=int(mcfg.get("lora_r", 8)),
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        dts_train_time = float(ckpt.get("train_time_sec", 0.0))
        LOG.info("Loaded DTS-GSSF checkpoint from %s", dts_ckpt)
    else:
        model, _ = dts.train_offline(bundle, wcfg, split, mcfg, tcfg, device, verbose=True)
        dts_train_time = time.time() - t0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_time_sec": dts_train_time,
                "mcfg": mcfg,
                "tcfg": dataclasses.asdict(tcfg),
            },
            dts_ckpt,
        )
    base_model = model
    y_val_pred_all = dts.predict_windows(bundle, base_model, wcfg, val_rng, device=device)[1]
    y_test_pred_all = dts.predict_windows(bundle, base_model, wcfg, test_rng, device=device)[1]

    if args.target_scope == "bottom":
        y_val_pred = y_val_pred_all[:, :, : bundle.y_bottom.shape[1]]
        y_test_pred = y_test_pred_all[:, :, : bundle.y_bottom.shape[1]]
    elif args.target_scope == "total":
        y_val_pred = y_val_pred_all[:, :, [-1]]
        y_test_pred = y_test_pred_all[:, :, [-1]]
    else:
        y_val_pred = y_val_pred_all
        y_test_pred = y_test_pred_all

    train_series = y_target[train_rng[0] : train_rng[1]]
    metrics_val = compute_metrics(y_val[:, 0], y_val_pred[:, 0], train_series, args.seasonality)
    metrics_test = compute_metrics(y_test[:, 0], y_test_pred[:, 0], train_series, args.seasonality)
    metrics_val_full = compute_metrics(
        flatten_horizon(y_val),
        flatten_horizon(y_val_pred),
        train_series,
        args.seasonality,
    )
    metrics_test_full = compute_metrics(
        flatten_horizon(y_test),
        flatten_horizon(y_test_pred),
        train_series,
        args.seasonality,
    )
    results["DTS-GSSF"] = {
        "status": "ok",
        "train_time_sec": dts_train_time,
        "val_h1": metrics_val,
        "test_h1": metrics_test,
        "val_full": metrics_val_full,
        "test_full": metrics_test_full,
    }
    persist_state()

    LOG.info("Running Seasonal Naive baseline")
    t0 = time.time()
    sn_val_pred = seasonal_naive_forecast(y_target, idxs_val, args.horizon, args.seasonality)
    sn_test_pred = seasonal_naive_forecast(y_target, idxs_test, args.horizon, args.seasonality)
    results["Seasonal Naive"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], sn_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], sn_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(sn_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(sn_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    LOG.info("Running Historical Average baseline")
    t0 = time.time()
    ha_val_pred = historical_average_forecast(
        y_target, bundle.time_index, idxs_val, args.horizon, train_rng
    )
    ha_test_pred = historical_average_forecast(
        y_target, bundle.time_index, idxs_test, args.horizon, train_rng
    )
    results["Historical Avg"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], ha_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], ha_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(ha_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(ha_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    LOG.info("Training LSTM Seq2Seq baseline")
    t0 = time.time()
    ds_train = TorchWindowDataset(X_rnn, y_rnn, wcfg, train_rng)
    ds_val = TorchWindowDataset(X_rnn, y_rnn, wcfg, val_rng)
    ds_test = TorchWindowDataset(X_rnn, y_rnn, wcfg, test_rng)
    input_dim = bundle.X.shape[1] * bundle.X.shape[2]
    lstm_ckpt = checkpoint_path("lstm_seq2seq")
    lstm = Seq2SeqRNN(
        input_dim=input_dim,
        hidden_dim=args.rnn_hidden,
        horizon=args.horizon,
        n_series=y_target.shape[1],
        rnn_type="lstm",
        num_layers=args.rnn_layers,
        dropout=args.rnn_dropout,
    )
    if args.resume and os.path.exists(lstm_ckpt):
        ckpt = torch.load(lstm_ckpt, map_location=device)
        lstm.load_state_dict(ckpt["model_state_dict"])
        lstm.to(device).eval()
        LOG.info("Loaded LSTM checkpoint from %s", lstm_ckpt)
    else:
        if args.tune_baselines:
            search = [
                {"hidden": args.rnn_hidden, "layers": args.rnn_layers, "dropout": args.rnn_dropout, "lr": 1e-3},
                {"hidden": args.rnn_hidden + 32, "layers": args.rnn_layers, "dropout": args.rnn_dropout, "lr": 5e-4},
                {"hidden": max(64, args.rnn_hidden - 32), "layers": max(1, args.rnn_layers - 1), "dropout": 0.2, "lr": 1e-3},
            ]
            def build(cfg):
                return Seq2SeqRNN(
                    input_dim=input_dim,
                    hidden_dim=cfg["hidden"],
                    horizon=args.horizon,
                    n_series=y_target.shape[1],
                    rnn_type="lstm",
                    num_layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )
            lstm, info = tune_torch_model(
                build,
                search,
                ds_train,
                ds_val,
                device,
                max(10, args.rnn_epochs // 2),
                args.batch_size,
                args.tune_max_trials,
                "LSTM Seq2Seq",
            )
            tuning_summary["LSTM Seq2Seq"] = info
        else:
            lstm = train_torch_model(
                lstm,
                ds_train,
                ds_val,
                device,
                args.rnn_epochs,
                1e-3,
                args.batch_size,
                checkpoint_path=lstm_ckpt,
                resume=args.resume,
                model_name="LSTM Seq2Seq",
                show_progress=True,
            )
    lstm_val_pred = predict_torch_model(lstm, ds_val, device, args.batch_size)
    lstm_test_pred = predict_torch_model(lstm, ds_test, device, args.batch_size)
    lstm_val_pred = y_scaler.inverse_transform(lstm_val_pred)
    lstm_test_pred = y_scaler.inverse_transform(lstm_test_pred)
    results["LSTM Seq2Seq"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], lstm_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], lstm_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(lstm_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(lstm_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    LOG.info("Training GRU Seq2Seq baseline")
    t0 = time.time()
    gru_ckpt = checkpoint_path("gru_seq2seq")
    gru = Seq2SeqRNN(
        input_dim=input_dim,
        hidden_dim=args.rnn_hidden,
        horizon=args.horizon,
        n_series=y_target.shape[1],
        rnn_type="gru",
        num_layers=args.rnn_layers,
        dropout=args.rnn_dropout,
    )
    if args.resume and os.path.exists(gru_ckpt):
        ckpt = torch.load(gru_ckpt, map_location=device)
        gru.load_state_dict(ckpt["model_state_dict"])
        gru.to(device).eval()
        LOG.info("Loaded GRU checkpoint from %s", gru_ckpt)
    else:
        if args.tune_baselines:
            search = [
                {"hidden": args.rnn_hidden, "layers": args.rnn_layers, "dropout": args.rnn_dropout, "lr": 1e-3},
                {"hidden": args.rnn_hidden + 32, "layers": args.rnn_layers, "dropout": args.rnn_dropout, "lr": 5e-4},
                {"hidden": max(64, args.rnn_hidden - 32), "layers": max(1, args.rnn_layers - 1), "dropout": 0.2, "lr": 1e-3},
            ]
            def build(cfg):
                return Seq2SeqRNN(
                    input_dim=input_dim,
                    hidden_dim=cfg["hidden"],
                    horizon=args.horizon,
                    n_series=y_target.shape[1],
                    rnn_type="gru",
                    num_layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )
            gru, info = tune_torch_model(
                build,
                search,
                ds_train,
                ds_val,
                device,
                max(10, args.rnn_epochs // 2),
                args.batch_size,
                args.tune_max_trials,
                "GRU Seq2Seq",
            )
            tuning_summary["GRU Seq2Seq"] = info
        else:
            gru = train_torch_model(
                gru,
                ds_train,
                ds_val,
                device,
                args.rnn_epochs,
                1e-3,
                args.batch_size,
                checkpoint_path=gru_ckpt,
                resume=args.resume,
                model_name="GRU Seq2Seq",
                show_progress=True,
            )
    gru_val_pred = predict_torch_model(gru, ds_val, device, args.batch_size)
    gru_test_pred = predict_torch_model(gru, ds_test, device, args.batch_size)
    gru_val_pred = y_scaler.inverse_transform(gru_val_pred)
    gru_test_pred = y_scaler.inverse_transform(gru_test_pred)
    results["GRU Seq2Seq"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], gru_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], gru_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(gru_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(gru_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    LOG.info("Training Transformer baseline")
    t0 = time.time()
    tx_ckpt = checkpoint_path("transformer")
    transformer = TransformerForecaster(
        input_dim=input_dim,
        d_model=args.tx_d_model,
        n_heads=args.tx_heads,
        num_layers=args.tx_layers,
        dim_feedforward=args.tx_ff,
        horizon=args.horizon,
        n_series=y_target.shape[1],
        dropout=args.tx_dropout,
    )
    if args.resume and os.path.exists(tx_ckpt):
        ckpt = torch.load(tx_ckpt, map_location=device)
        transformer.load_state_dict(ckpt["model_state_dict"])
        transformer.to(device).eval()
        LOG.info("Loaded Transformer checkpoint from %s", tx_ckpt)
    else:
        if args.tune_baselines:
            search = [
                {"d_model": args.tx_d_model, "heads": args.tx_heads, "layers": args.tx_layers, "ff": args.tx_ff, "dropout": args.tx_dropout, "lr": 1e-3},
                {"d_model": args.tx_d_model + 64, "heads": args.tx_heads, "layers": max(2, args.tx_layers - 1), "ff": args.tx_ff + 128, "dropout": args.tx_dropout, "lr": 5e-4},
                {"d_model": max(64, args.tx_d_model - 32), "heads": args.tx_heads, "layers": args.tx_layers, "ff": max(128, args.tx_ff - 64), "dropout": 0.2, "lr": 1e-3},
            ]
            def build(cfg):
                return TransformerForecaster(
                    input_dim=input_dim,
                    d_model=cfg["d_model"],
                    n_heads=cfg["heads"],
                    num_layers=cfg["layers"],
                    dim_feedforward=cfg["ff"],
                    horizon=args.horizon,
                    n_series=y_target.shape[1],
                    dropout=cfg["dropout"],
                )
            transformer, info = tune_torch_model(
                build,
                search,
                ds_train,
                ds_val,
                device,
                max(10, args.rnn_epochs // 2),
                args.batch_size,
                args.tune_max_trials,
                "Transformer",
            )
            tuning_summary["Transformer"] = info
        else:
            transformer = train_torch_model(
                transformer,
                ds_train,
                ds_val,
                device,
                args.rnn_epochs,
                1e-3,
                args.batch_size,
                checkpoint_path=tx_ckpt,
                resume=args.resume,
                model_name="Transformer",
                show_progress=True,
            )
    tx_val_pred = predict_torch_model(transformer, ds_val, device, args.batch_size)
    tx_test_pred = predict_torch_model(transformer, ds_test, device, args.batch_size)
    tx_val_pred = y_scaler.inverse_transform(tx_val_pred)
    tx_test_pred = y_scaler.inverse_transform(tx_test_pred)
    results["Transformer"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], tx_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], tx_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(tx_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(tx_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    LOG.info("Training DCRNN baseline")
    t0 = time.time()
    dcrnn_ckpt = checkpoint_path("dcrnn")
    dcrnn = DCRNNBaseline(
        n_nodes=bundle.X.shape[1],
        f_in=bundle.X.shape[2],
        hidden_dim=args.dcrnn_hidden,
        horizon=args.horizon,
        n_series=y_target.shape[1],
        A_phys=bundle.net.A_phys,
        num_layers=args.dcrnn_layers,
    )
    if args.resume and os.path.exists(dcrnn_ckpt):
        ckpt = torch.load(dcrnn_ckpt, map_location=device)
        dcrnn.load_state_dict(ckpt["model_state_dict"])
        dcrnn.to(device).eval()
        LOG.info("Loaded DCRNN checkpoint from %s", dcrnn_ckpt)
    else:
        if args.tune_baselines:
            search = [
                {"hidden": args.dcrnn_hidden, "layers": args.dcrnn_layers, "lr": 1e-3},
                {"hidden": args.dcrnn_hidden + 16, "layers": args.dcrnn_layers, "lr": 5e-4},
                {"hidden": max(32, args.dcrnn_hidden - 16), "layers": max(1, args.dcrnn_layers - 1), "lr": 1e-3},
            ]
            def build(cfg):
                return DCRNNBaseline(
                    n_nodes=bundle.X.shape[1],
                    f_in=bundle.X.shape[2],
                    hidden_dim=cfg["hidden"],
                    horizon=args.horizon,
                    n_series=y_target.shape[1],
                    A_phys=bundle.net.A_phys,
                    num_layers=cfg["layers"],
                )
            dcrnn, info = tune_torch_model(
                build,
                search,
                ds_train,
                ds_val,
                device,
                max(10, args.rnn_epochs // 2),
                args.batch_size,
                args.tune_max_trials,
                "DCRNN",
            )
            tuning_summary["DCRNN"] = info
        else:
            dcrnn = train_torch_model(
                dcrnn,
                ds_train,
                ds_val,
                device,
                args.rnn_epochs,
                1e-3,
                args.batch_size,
                checkpoint_path=dcrnn_ckpt,
                resume=args.resume,
                model_name="DCRNN",
                show_progress=True,
            )
    dcrnn_val_pred = predict_torch_model(dcrnn, ds_val, device, args.batch_size)
    dcrnn_test_pred = predict_torch_model(dcrnn, ds_test, device, args.batch_size)
    dcrnn_val_pred = y_scaler.inverse_transform(dcrnn_val_pred)
    dcrnn_test_pred = y_scaler.inverse_transform(dcrnn_test_pred)
    results["DCRNN"] = {
        "status": "ok",
        "train_time_sec": time.time() - t0,
        "val_h1": compute_metrics(y_val[:, 0], dcrnn_val_pred[:, 0], train_series, args.seasonality),
        "test_h1": compute_metrics(y_test[:, 0], dcrnn_test_pred[:, 0], train_series, args.seasonality),
        "val_full": compute_metrics(
            flatten_horizon(y_val),
            flatten_horizon(dcrnn_val_pred),
            train_series,
            args.seasonality,
        ),
        "test_full": compute_metrics(
            flatten_horizon(y_test),
            flatten_horizon(dcrnn_test_pred),
            train_series,
            args.seasonality,
        ),
    }
    persist_state()

    ablation_results = {}
    if args.run_ablations:
        LOG.info("Running ablation study")
        ablation_cfgs = [
            ("No LoRA", {"lora_r": 0, "use_graph": True, "use_adaptive": True}),
            ("No Adaptive Adj", {"lora_r": args.lora_r, "use_graph": True, "use_adaptive": False}),
            ("No Graph", {"lora_r": args.lora_r, "use_graph": False, "use_adaptive": False}),
        ]
        for name, cfg in ablation_cfgs:
            LOG.info("Ablation: %s", name)
            model = DTSGSSF_Ablation(
                N=bundle.X.shape[1],
                F_in=bundle.X.shape[2],
                n_series=bundle.y_all.shape[1],
                n_agg=bundle.y_all.shape[1] - bundle.X.shape[1],
                A_phys=bundle.net.A_phys,
                d_model=args.d_model,
                horizon=args.horizon,
                K=args.K,
                lora_r=cfg["lora_r"],
                dropout=args.dropout,
                use_graph=cfg["use_graph"],
                use_adaptive=cfg["use_adaptive"],
            )
            tcfg_ab = dts.TrainConfig(
                epochs=args.ablation_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                early_stopping_patience=args.early_patience,
                accumulation_steps=args.accum_steps,
                grad_clip=args.grad_clip,
            )
            model = train_custom_dts(bundle, wcfg, split, model, tcfg_ab, device, verbose=False)
            y_val_pred_all = dts.predict_windows(bundle, model, wcfg, val_rng, device=device)[1]
            y_test_pred_all = dts.predict_windows(bundle, model, wcfg, test_rng, device=device)[1]
            if args.target_scope == "bottom":
                y_val_pred = y_val_pred_all[:, :, : bundle.y_bottom.shape[1]]
                y_test_pred = y_test_pred_all[:, :, : bundle.y_bottom.shape[1]]
            elif args.target_scope == "total":
                y_val_pred = y_val_pred_all[:, :, [-1]]
                y_test_pred = y_test_pred_all[:, :, [-1]]
            else:
                y_val_pred = y_val_pred_all
                y_test_pred = y_test_pred_all
            ablation_results[name] = {
                "val_h1": compute_metrics(y_val[:, 0], y_val_pred[:, 0], train_series, args.seasonality),
                "test_h1": compute_metrics(y_test[:, 0], y_test_pred[:, 0], train_series, args.seasonality),
                "val_full": compute_metrics(
                    flatten_horizon(y_val),
                    flatten_horizon(y_val_pred),
                    train_series,
                    args.seasonality,
                ),
                "test_full": compute_metrics(
                    flatten_horizon(y_test),
                    flatten_horizon(y_test_pred),
                    train_series,
                    args.seasonality,
                ),
                "config": cfg,
            }
        save_json(os.path.join(out_dir, "ablations_summary.json"), np_to_py(ablation_results))
        try:
            LOG.info("Running online ablations (residual/reconciliation/drift)")
            ocfg = dts.OnlineConfig(adapt_steps=18)
            res_full = dts.online_run(bundle, base_model, wcfg, split, ocfg, device)
            ocfg_no_adapt = dts.OnlineConfig(adapt_steps=0)
            res_no_adapt = dts.online_run(bundle, base_model, wcfg, split, ocfg_no_adapt, device)
            N = bundle.y_bottom.shape[1]
            online = {
                "base": {
                    "mae": dts.mae_np(res_full.y_true[:, :N], res_full.y_base[:, :N]),
                },
                "kalman": {
                    "mae": dts.mae_np(res_full.y_true[:, :N], res_full.y_corr[:, :N]),
                },
                "kalman_recon": {
                    "mae": dts.mae_np(res_full.y_true[:, :N], res_full.y_recon[:, :N]),
                },
                "no_adapt_kalman": {
                    "mae": dts.mae_np(res_no_adapt.y_true[:, :N], res_no_adapt.y_corr[:, :N]),
                },
                "no_adapt_kalman_recon": {
                    "mae": dts.mae_np(res_no_adapt.y_true[:, :N], res_no_adapt.y_recon[:, :N]),
                },
            }
            save_json(os.path.join(out_dir, "online_ablations.json"), np_to_py(online))
        except Exception as exc:
            LOG.warning("Online ablations failed: %s", exc)

    if not args.skip_sarimax:
        if has_result("SARIMAX"):
            LOG.info("Skipping SARIMAX (already completed)")
        else:
            LOG.info("Running SARIMAX baseline (horizon=1)")
            t0 = time.time()
            try:
                sarimax_val_pred = sarimax_forecast_h1(
                    y_target,
                    idxs_val,
                    1,
                    args.seasonality,
                    train_rng,
                    args.sarimax_series_limit,
                    args.sarimax_maxiter,
                    args.sarimax_train_limit,
                    "val",
                )
                sarimax_test_pred = sarimax_forecast_h1(
                    y_target,
                    idxs_test,
                    1,
                    args.seasonality,
                    train_rng,
                    args.sarimax_series_limit,
                    args.sarimax_maxiter,
                    args.sarimax_train_limit,
                    "test",
                )
                results["SARIMAX"] = {
                    "status": "ok",
                    "train_time_sec": time.time() - t0,
                    "val_h1": compute_metrics(y_val[:, 0], sarimax_val_pred[:, 0], train_series, args.seasonality),
                    "test_h1": compute_metrics(y_test[:, 0], sarimax_test_pred[:, 0], train_series, args.seasonality),
                }
            except Exception as exc:
                results["SARIMAX"] = {"status": "skipped", "reason": str(exc)}
            persist_state()

    if not args.skip_xgboost:
        if has_result("XGBoost"):
            LOG.info("Skipping XGBoost (already completed)")
        else:
            LOG.info("Running XGBoost baseline (horizon=1)")
            t0 = time.time()
            try:
                xgb_val_pred = xgboost_forecast_h1(
                    bundle.X, y_target, wcfg, idxs_val, train_rng, args.xgb_series_limit
                )
                xgb_test_pred = xgboost_forecast_h1(
                    bundle.X, y_target, wcfg, idxs_test, train_rng, args.xgb_series_limit
                )
                results["XGBoost"] = {
                    "status": "ok",
                    "train_time_sec": time.time() - t0,
                    "val_h1": compute_metrics(y_val[:, 0], xgb_val_pred[:, 0], train_series, args.seasonality),
                    "test_h1": compute_metrics(y_test[:, 0], xgb_test_pred[:, 0], train_series, args.seasonality),
                }
            except Exception as exc:
                results["XGBoost"] = {"status": "skipped", "reason": str(exc)}
            persist_state()

    preq_metrics = {}
    latency_metrics = {}
    if args.run_prequential:
        LOG.info("Running prequential evaluation")
        y_true_p, y_pred_p, lat = prequential_eval_model(
            base_model,
            bundle.X,
            y_target,
            wcfg,
            test_rng,
            device,
        )
        preq_metrics["DTS-GSSF"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["DTS-GSSF"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            None,
            y_target,
            y_target,
            wcfg,
            test_rng,
            device,
            baseline="seasonal_naive",
            seasonality=args.seasonality,
        )
        preq_metrics["Seasonal Naive"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["Seasonal Naive"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            None,
            y_target,
            y_target,
            wcfg,
            test_rng,
            device,
            baseline="historical_avg",
            time_index=bundle.time_index,
            train_rng=train_rng,
            seasonality=args.seasonality,
        )
        preq_metrics["Historical Avg"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["Historical Avg"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            lstm,
            X_rnn,
            y_rnn,
            wcfg,
            test_rng,
            device,
            inverse_fn=y_scaler.inverse_transform,
        )
        preq_metrics["LSTM Seq2Seq"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["LSTM Seq2Seq"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            gru,
            X_rnn,
            y_rnn,
            wcfg,
            test_rng,
            device,
            inverse_fn=y_scaler.inverse_transform,
        )
        preq_metrics["GRU Seq2Seq"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["GRU Seq2Seq"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            transformer,
            X_rnn,
            y_rnn,
            wcfg,
            test_rng,
            device,
            inverse_fn=y_scaler.inverse_transform,
        )
        preq_metrics["Transformer"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["Transformer"] = lat

        y_true_p, y_pred_p, lat = prequential_eval_model(
            dcrnn,
            X_rnn,
            y_rnn,
            wcfg,
            test_rng,
            device,
            inverse_fn=y_scaler.inverse_transform,
        )
        preq_metrics["DCRNN"] = compute_metrics(y_true_p, y_pred_p, train_series, args.seasonality)
        latency_metrics["DCRNN"] = lat

        save_json(os.path.join(out_dir, "prequential_metrics.json"), np_to_py(preq_metrics))
        save_json(os.path.join(out_dir, "latency_metrics.json"), np_to_py(latency_metrics))

    preds_test_full = {
        "DTS-GSSF": y_test_pred,
        "Seasonal Naive": sn_test_pred,
        "Historical Avg": ha_test_pred,
        "LSTM Seq2Seq": lstm_test_pred,
        "GRU Seq2Seq": gru_test_pred,
        "Transformer": tx_test_pred,
        "DCRNN": dcrnn_test_pred,
    }
    preds_test_h1 = {k: v[:, 0] for k, v in preds_test_full.items()}

    if args.bootstrap_samples > 0:
        LOG.info("Running bootstrap CIs (%d samples)", args.bootstrap_samples)
        ci = {}
        delta = {}
        for name, pred in preds_test_full.items():
            ci[name] = {
                "mae": bootstrap_ci(
                    flatten_horizon(y_test),
                    flatten_horizon(pred),
                    lambda a, b: float(np.mean(np.abs(a - b))),
                    args.bootstrap_samples,
                    args.seed,
                ),
                "wape": bootstrap_ci(
                    flatten_horizon(y_test),
                    flatten_horizon(pred),
                    lambda a, b: float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + 1e-8) * 100.0),
                    args.bootstrap_samples,
                    args.seed,
                ),
            }
        if "DTS-GSSF" in preds_test_full:
            base_pred = preds_test_full["DTS-GSSF"]
            for name, pred in preds_test_full.items():
                if name == "DTS-GSSF":
                    continue
                delta[name] = paired_bootstrap_delta(
                    flatten_horizon(y_test),
                    flatten_horizon(pred),
                    flatten_horizon(base_pred),
                    lambda a, b: float(np.mean(np.abs(a - b))),
                    args.bootstrap_samples,
                    args.seed,
                )
        save_json(os.path.join(out_dir, "bootstrap_ci.json"), np_to_py(ci))
        save_json(os.path.join(out_dir, "bootstrap_deltas.json"), np_to_py(delta))

    metric_keys = ["mae", "rmse", "mse", "mape", "smape", "wape", "accuracy_wape", "r2", "mase"]
    val_rows = []
    test_rows = []
    val_full_rows = []
    test_full_rows = []
    for name, payload in results.items():
        if payload.get("status") != "ok":
            continue
        val = payload.get("val_h1", {})
        test = payload.get("test_h1", {})
        val_full = payload.get("val_full", {})
        test_full = payload.get("test_full", {})
        val_rows.append({"model": name, **val})
        test_rows.append({"model": name, **test})
        if val_full:
            val_full_rows.append({"model": name, **val_full})
        if test_full:
            test_full_rows.append({"model": name, **test_full})

    val_df = build_metric_table(val_rows, metric_keys)
    test_df = build_metric_table(test_rows, metric_keys)
    val_full_df = build_metric_table(val_full_rows, metric_keys) if val_full_rows else None
    test_full_df = build_metric_table(test_full_rows, metric_keys) if test_full_rows else None

    val_df.to_csv(os.path.join(out_dir, "metrics_val_h1.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "metrics_test_h1.csv"), index=False)
    if val_full_df is not None:
        val_full_df.to_csv(os.path.join(out_dir, "metrics_val_full.csv"), index=False)
    if test_full_df is not None:
        test_full_df.to_csv(os.path.join(out_dir, "metrics_test_full.csv"), index=False)
    save_metric_charts(test_df, charts_dir, "test_h1")
    if test_full_df is not None:
        save_metric_charts(test_full_df, charts_dir, "test_full")

    summary = {
        "config": dataclasses.asdict(cfg),
        "window_config": dataclasses.asdict(wcfg),
        "split_config": dataclasses.asdict(split),
        "target_scope": args.target_scope,
        "reproducibility": {
            "ssm_type": "GatedSSMBlock",
            "graph_type": "GraphPropagation",
            "lora_alpha": 16.0,
            "kalman_init": "P ~ N(0,1), row-normalized; F=I*F_decay; Q=I*q; R=I*r_scale",
            "drift_detector": "PageHinkley",
        },
        "train_config": dataclasses.asdict(tcfg),
        "model_config": mcfg,
        "baseline_config": {
            "rnn_hidden": args.rnn_hidden,
            "rnn_layers": args.rnn_layers,
            "rnn_dropout": args.rnn_dropout,
            "transformer": {
                "d_model": args.tx_d_model,
                "heads": args.tx_heads,
                "layers": args.tx_layers,
                "ff": args.tx_ff,
                "dropout": args.tx_dropout,
            },
            "dcrnn": {
                "hidden": args.dcrnn_hidden,
                "layers": args.dcrnn_layers,
            },
        },
        "scaling": {
            "x_standardized": True,
            "y_standardized": True,
        },
        "tuning_summary": tuning_summary,
        "results": results,
    }
    save_json(os.path.join(out_dir, "metrics_summary.json"), np_to_py(summary))

    sample_series = 0
    if y_target.shape[1] > 0:
        preds_for_plot = {
            "DTS-GSSF": y_test_pred[:, 0, sample_series],
            "Seasonal Naive": sn_test_pred[:, 0, sample_series],
            "Historical Avg": ha_test_pred[:, 0, sample_series],
        }
        save_forecast_chart(
            bundle.time_index,
            y_target[idxs_test, sample_series],
            preds_for_plot,
            idxs_test,
            os.path.join(charts_dir, "forecast_sample.html"),
            target_names[sample_series],
        )

    LOG.info("Saved outputs to %s", out_dir)


if __name__ == "__main__":
    main()

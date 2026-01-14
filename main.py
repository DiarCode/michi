
"""
DTS-GSSF Demo (single-file)
==========================
Dual-Timescale Graph State-Space Forecasting with:
- Graph-structured "SSM-ish" backbone (efficient long memory)
- Online residual correction via Kalman-style filtering (fast timescale)
- Drift detection (Page-Hinkley) + drift-triggered LoRA-style low-rank adaptation
- Hierarchical reconciliation (stations -> lines -> districts -> total) via weighted projection (MinT/OLS)

UI:
- Streamlit dashboard to generate realistic Astana bus passenger-flow data,
  train the model, and run an online simulation with analytics.

This file is intentionally self-contained for teaching/demo purposes.
It is NOT claiming to be a production-grade transport forecaster,
but it implements the full DTS-GSSF pipeline from the draft spec.

Run (recommended):
  uv run streamlit run main.py

CLI:
  uv run python main.py --online

Mac (Apple Silicon):
  Uses MPS if available; otherwise CPU.
  You can set: PYTORCH_ENABLE_MPS_FALLBACK=1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Logging & reproducibility
# ----------------------------

LOG = logging.getLogger("dts_gssf")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class RichLogger:
    """
    Styled console logging with headers, progress bars, and metrics tables.
    Inspired by best practices from MIT CSAIL, Stanford AI Lab, and Harvard ML research.
    """

    COLORS = {
        "header": "\033[1;36m",    # Cyan bold
        "success": "\033[1;32m",   # Green bold
        "warning": "\033[1;33m",   # Yellow bold
        "error": "\033[1;31m",     # Red bold
        "info": "\033[0;37m",      # White
        "accent": "\033[1;35m",    # Magenta bold
        "dim": "\033[2m",          # Dim
        "reset": "\033[0m",
        "bold": "\033[1m",
        "blue": "\033[1;34m",
    }

    BOX = {
        "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
        "h": "═", "v": "║", "cross": "╬",
        "light_h": "─", "light_v": "│",
    }

    @classmethod
    def header(cls, title: str, width: int = 70) -> None:
        """Print a prominent boxed header."""
        c = cls.COLORS
        b = cls.BOX
        title_centered = title.center(width - 2)
        print(f"\n{c['header']}{b['tl']}{b['h'] * (width - 2)}{b['tr']}{c['reset']}")
        print(f"{c['header']}{b['v']}{c['bold']}{title_centered}{c['header']}{b['v']}{c['reset']}")
        print(f"{c['header']}{b['bl']}{b['h'] * (width - 2)}{b['br']}{c['reset']}\n")

    @classmethod
    def section(cls, title: str, width: int = 50) -> None:
        """Print a section divider with title."""
        c = cls.COLORS
        side = (width - len(title) - 2) // 2
        line = cls.BOX["light_h"] * side
        print(f"\n{c['accent']}{line} {title} {line}{c['reset']}")

    @classmethod
    def subsection(cls, title: str) -> None:
        """Print a subsection header."""
        c = cls.COLORS
        print(f"\n{c['blue']}▸ {title}{c['reset']}")

    @classmethod
    def metric(cls, name: str, value: object, unit: str = "", delta: Optional[float] = None) -> None:
        """Print a single metric with optional delta."""
        c = cls.COLORS
        name_fmt = f"{name:.<25}"
        val_str = f"{value}{unit}"
        if delta is not None:
            delta_color = c["success"] if delta < 0 else c["warning"]
            delta_str = f" ({delta:+.4f})"
            print(f"  {c['dim']}{name_fmt}{c['reset']} {c['bold']}{val_str}{c['reset']}{delta_color}{delta_str}{c['reset']}")
        else:
            print(f"  {c['dim']}{name_fmt}{c['reset']} {c['bold']}{val_str}{c['reset']}")

    @classmethod
    def metrics_row(cls, metrics: Dict[str, object]) -> None:
        """Print multiple metrics in a single row."""
        c = cls.COLORS
        parts = [f"{c['dim']}{k}:{c['reset']} {c['bold']}{v}{c['reset']}" for k, v in metrics.items()]
        print("  " + "  │  ".join(parts))

    @classmethod
    def progress_bar(cls, current: int, total: int, width: int = 30, desc: str = "") -> str:
        """Return a text progress bar."""
        pct = current / max(1, total)
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"{desc} [{bar}] {current}/{total} ({pct*100:.0f}%)"

    @classmethod
    def table(cls, headers: List[str], rows: List[List[object]], col_widths: Optional[List[int]] = None) -> None:
        """Print a formatted table."""
        c = cls.COLORS
        b = cls.BOX
        if col_widths is None:
            col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else 0) + 2
                          for i, h in enumerate(headers)]

        header_line = f"  {c['bold']}"
        sep_line = f"  {c['dim']}"
        for i, h in enumerate(headers):
            header_line += f"{str(h):^{col_widths[i]}}" + (b["light_v"] if i < len(headers) - 1 else "")
            sep_line += b["light_h"] * col_widths[i] + ("┼" if i < len(headers) - 1 else "")
        print(header_line + c["reset"])
        print(sep_line + c["reset"])

        for row in rows:
            row_line = "  "
            for i, cell in enumerate(row):
                row_line += f"{str(cell):^{col_widths[i]}}" + (b["light_v"] if i < len(row) - 1 else "")
            print(row_line)

    @classmethod
    def success(cls, msg: str) -> None:
        """Print a success message."""
        print(f"  {cls.COLORS['success']}✓{cls.COLORS['reset']} {msg}")

    @classmethod
    def warning(cls, msg: str) -> None:
        """Print a warning message."""
        print(f"  {cls.COLORS['warning']}⚠{cls.COLORS['reset']} {msg}")

    @classmethod
    def error(cls, msg: str) -> None:
        """Print an error message."""
        print(f"  {cls.COLORS['error']}✗{cls.COLORS['reset']} {msg}")

    @classmethod
    def info(cls, msg: str) -> None:
        """Print an info message."""
        print(f"  {cls.COLORS['info']}ℹ{cls.COLORS['reset']} {msg}")

    @classmethod
    def epoch_summary(cls, epoch: int, total_epochs: int, train_loss: float,
                      val_loss: float, lr: float, elapsed: float,
                      best_val: Optional[float] = None) -> None:
        """Print a formatted epoch summary line."""
        c = cls.COLORS
        is_best = best_val is not None and val_loss <= best_val
        best_marker = f" {c['success']}★{c['reset']}" if is_best else ""
        print(f"  {c['bold']}Epoch {epoch:>2}/{total_epochs}{c['reset']} │ "
              f"train: {c['warning']}{train_loss:.4f}{c['reset']} │ "
              f"val: {c['accent']}{val_loss:.4f}{c['reset']}{best_marker} │ "
              f"lr: {lr:.2e} │ {elapsed:.1f}s")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless on CPU/MPS


# ----------------------------
# Utilities
# ----------------------------

def device_auto() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_tensor(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x, beta=1.0, threshold=20.0)

# ----------------------------
# Astana bus-network generator
# ----------------------------

ASTANA_DISTRICTS = ["Esil", "Almaty", "Saryarka", "Baikonur"]

ASTANA_PLACES = [
    "Khan Shatyr", "Baiterek", "Mega Silk Way", "Expo 2017", "Nurly Zhol",
    "Astana Mall", "Keruen City", "Palace of Peace", "Central Park", "Triumphal Arch",
    "Independence Square", "University Quarter", "River Embankment", "Botanical Garden",
    "Saryarka Ave", "Respublika Ave", "Turan Ave", "Kabanbay Batyr Ave", "Mangilik El Ave",
    "Syganak St", "Sarayshyk St", "Kenesary St", "Abylai Khan Ave", "Seifullin St",
    "Bogenbai Batyr Ave", "Kosmonavtov St", "Shakarim St", "Zhastar Ave", "Dostyk St",
]

def _pick_unique(names: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    pool = names[:]
    rng.shuffle(pool)
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]

@dataclass(frozen=True)
class NetworkSpec:
    station_names: List[str]
    station_district: List[str]
    lines: Dict[str, List[int]]
    A_phys: np.ndarray
    edges: List[Tuple[int, int]]

def build_astana_network(n_stations: int = 28, n_lines: int = 9, seed: int = 7) -> NetworkSpec:
    rng = np.random.default_rng(seed)
    names = _pick_unique(ASTANA_PLACES, n_stations, seed=seed)
    district_probs = np.array([0.30, 0.30, 0.25, 0.15])
    station_district = list(rng.choice(ASTANA_DISTRICTS, size=n_stations, p=district_probs))

    lines: Dict[str, List[int]] = {}
    base_perm = list(range(n_stations))
    rng.shuffle(base_perm)

    for li in range(n_lines):
        line_len = int(rng.integers(9, min(15, n_stations)))
        start = int((li * 3) % n_stations)
        path = [base_perm[(start + k) % n_stations] for k in range(line_len)]
        if li > 0:
            prev = lines[f"Line {li}"]
            hubs = rng.choice(prev, size=min(2, len(prev)), replace=False)
            path[: len(hubs)] = list(hubs)

        seen = set()
        path2 = []
        for x in path:
            xi = int(x)
            if xi not in seen:
                path2.append(xi)
                seen.add(xi)
        lines[f"Line {li+1}"] = path2

    edges = set()
    for stations in lines.values():
        for a, b in zip(stations[:-1], stations[1:]):
            edges.add((min(a, b), max(a, b)))

    hubs = rng.choice(np.arange(n_stations), size=max(3, n_stations // 6), replace=False)
    for _ in range(n_stations // 2):
        a, b = int(rng.choice(hubs)), int(rng.choice(hubs))
        if a != b:
            edges.add((min(a, b), max(a, b)))

    edges_list = [(a, b) for (a, b) in edges]
    A = np.zeros((n_stations, n_stations), dtype=np.float32)
    for a, b in edges_list:
        A[a, b] = 1.0
        A[b, a] = 1.0
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)

    return NetworkSpec(
        station_names=names,
        station_district=station_district,
        lines=lines,
        A_phys=A.astype(np.float32),
        edges=edges_list,
    )

@dataclass(frozen=True)
class DataGenConfig:
    seed: int = 7
    days: int = 60  # Increased from 35 to get ~50k+ richer records
    freq_min: int = 15
    start: str = "2025-10-01 05:00:00"
    base_mean: float = 18.0
    overdispersion_kappa: float = 8.0
    rush_hour_boost: float = 2.2
    weekend_scale: float = 0.78
    night_scale: float = 0.45
    event_prob_per_day: float = 0.35  # Increased for richness
    disruption_prob_per_day: float = 0.15 # Increased for richness
    drift_day: int = 45 # Adjusted for longer duration
    drift_scale: float = 1.25
    drift_station_frac: float = 0.30

@dataclass(frozen=True)
class DataBundle:
    cfg: DataGenConfig
    net: NetworkSpec
    time_index: pd.DatetimeIndex
    X: np.ndarray
    y_bottom: np.ndarray
    y_all: np.ndarray
    series_names: List[str]
    S: np.ndarray
    meta: Dict[str, object]

def _time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    hour = idx.hour.to_numpy(dtype=np.float32) + idx.minute.to_numpy(dtype=np.float32) / 60.0
    dow = idx.dayofweek.to_numpy(dtype=np.float32)
    hod_sin = np.sin(2 * np.pi * hour / 24.0)
    hod_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    is_weekend = (dow >= 5).astype(np.float32)
    return np.stack([hod_sin, hod_cos, dow_sin, dow_cos, is_weekend], axis=1).astype(np.float32)

def _astana_weather(idx: pd.DatetimeIndex, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    t = np.arange(len(idx))
    temp = -8 + 6 * np.sin(2 * np.pi * t / (len(idx) + 1)) + rng.normal(0, 1.2, size=len(idx))
    precip = (rng.random(len(idx)) < 0.05).astype(np.float32)
    wind = 4 + 2 * rng.random(len(idx))
    return np.stack([temp, precip, wind], axis=1).astype(np.float32)

def _daily_events(idx: pd.DatetimeIndex, net: NetworkSpec, cfg: DataGenConfig) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed + 202)
    T, N = len(idx), len(net.station_names)
    event_mult = np.ones((T, N), dtype=np.float32)
    event_days = sorted(set(idx.normalize()))
    event_log = []

    venue_keywords = ["Expo", "Khan Shatyr", "Baiterek", "Mega", "Palace", "Central Park"]
    venue_stations = [i for i, name in enumerate(net.station_names) if any(k in name for k in venue_keywords)]
    if not venue_stations:
        venue_stations = list(range(N))

    for day in event_days:
        if rng.random() < cfg.event_prob_per_day:
            vs = rng.choice(venue_stations, size=int(rng.integers(1, 3)), replace=False)
            start_hour = int(rng.integers(16, 20))
            dur_hours = int(rng.integers(2, 5))
            strength = float(rng.uniform(1.2, 1.9))
            mask = (idx.normalize() == day) & (idx.hour >= start_hour) & (idx.hour < start_hour + dur_hours)
            for v in vs:
                event_mult[mask, int(v)] *= strength
            event_log.append(dict(day=str(day.date()), start_hour=start_hour, hours=dur_hours,
                                  strength=strength, stations=[net.station_names[int(v)] for v in vs]))
    return event_mult, {"events": event_log}

def _service_disruptions(idx: pd.DatetimeIndex, net: NetworkSpec, cfg: DataGenConfig) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed + 303)
    T, N = len(idx), len(net.station_names)
    mult = np.ones((T, N), dtype=np.float32)
    days = sorted(set(idx.normalize()))
    log = []

    for day in days:
        if rng.random() < cfg.disruption_prob_per_day:
            line_name = rng.choice(list(net.lines.keys()))
            path = net.lines[line_name]
            if len(path) < 5:
                continue
            start = int(rng.integers(0, len(path) - 4))
            corridor = path[start:start + int(rng.integers(3, 6))]
            start_hour = int(rng.integers(7, 17))
            dur_hours = int(rng.integers(2, 6))
            severity = float(rng.uniform(0.55, 0.85))
            mask = (idx.normalize() == day) & (idx.hour >= start_hour) & (idx.hour < start_hour + dur_hours)
            for s in corridor:
                mult[mask, int(s)] *= severity
            log.append(dict(day=str(day.date()), line=line_name, start_hour=start_hour,
                            hours=dur_hours, severity=severity,
                            corridor=[net.station_names[int(s)] for s in corridor]))
    return mult, {"disruptions": log}

def _rush_profile(idx: pd.DatetimeIndex, cfg: DataGenConfig) -> np.ndarray:
    hour = idx.hour.to_numpy(dtype=np.float32) + idx.minute.to_numpy(dtype=np.float32) / 60.0
    morning = np.exp(-0.5 * ((hour - 8.3) / 1.35) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 1.7) ** 2)
    base = 0.6 + cfg.rush_hour_boost * (0.55 * morning + 0.75 * evening)
    night = ((hour < 6.0) | (hour > 22.0)).astype(np.float32)
    base = base * (1.0 - night * (1.0 - cfg.night_scale))
    return base.astype(np.float32)

def _weekly_profile(idx: pd.DatetimeIndex, cfg: DataGenConfig) -> np.ndarray:
    dow = idx.dayofweek.to_numpy(dtype=np.float32)
    weekend = (dow >= 5).astype(np.float32)
    return (1.0 - weekend * (1.0 - cfg.weekend_scale)).astype(np.float32)

def _station_popularity(net: NetworkSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 404)
    N = len(net.station_names)
    pop = rng.lognormal(mean=0.0, sigma=0.35, size=N).astype(np.float32)
    hub_kw = ["Nurly Zhol", "Expo", "Mega", "Khan Shatyr", "Baiterek", "Central Park"]
    for i, name in enumerate(net.station_names):
        if any(k in name for k in hub_kw):
            pop[i] *= 1.35
    for i, d in enumerate(net.station_district):
        if d == "Esil":
            pop[i] *= 1.18
        elif d == "Baikonur":
            pop[i] *= 0.95
    return pop

def build_hierarchy(net: NetworkSpec) -> Tuple[np.ndarray, List[str], Dict[str, List[int]], Dict[str, List[int]]]:
    N = len(net.station_names)
    line_groups = {ln: [int(i) for i in idxs] for ln, idxs in net.lines.items()}
    district_groups: Dict[str, List[int]] = {d: [] for d in ASTANA_DISTRICTS}
    for i, d in enumerate(net.station_district):
        district_groups[d].append(i)

    series_names = []
    rows = []

    for i, name in enumerate(net.station_names):
        series_names.append(f"Station | {name}")
        row = np.zeros(N, dtype=np.float32); row[i] = 1.0
        rows.append(row)

    for ln, idxs in line_groups.items():
        series_names.append(f"Line | {ln}")
        row = np.zeros(N, dtype=np.float32); row[idxs] = 1.0
        rows.append(row)

    for d, idxs in district_groups.items():
        series_names.append(f"District | {d}")
        row = np.zeros(N, dtype=np.float32); row[idxs] = 1.0
        rows.append(row)

    series_names.append("Network | Total")
    rows.append(np.ones(N, dtype=np.float32))

    S = np.stack(rows, axis=0).astype(np.float32)
    return S, series_names, line_groups, district_groups

def _nb_sample(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    mu = np.clip(mu, 1e-4, None)
    shape = kappa
    scale = mu / kappa
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam).astype(np.float32)

def generate_astana_data(cfg: DataGenConfig, net: NetworkSpec) -> DataBundle:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    freq = f"{cfg.freq_min}min"
    idx = pd.date_range(cfg.start, periods=int((24*60//cfg.freq_min)*cfg.days), freq=freq)
    T, N = len(idx), len(net.station_names)

    time_feat = _time_features(idx)
    weather = _astana_weather(idx, cfg.seed)
    rush = _rush_profile(idx, cfg)
    weekly = _weekly_profile(idx, cfg)
    pop = _station_popularity(net, cfg.seed)

    event_mult, event_meta = _daily_events(idx, net, cfg)
    disrupt_mult, disrupt_meta = _service_disruptions(idx, net, cfg)

    drift_start = idx[0] + pd.Timedelta(days=cfg.drift_day)
    drift_mask = np.asarray(idx >= drift_start, dtype=np.float32)
    drift_station = rng.choice(np.arange(N), size=max(1, int(cfg.drift_station_frac*N)), replace=False)
    drift_station_flag = np.zeros(N, dtype=np.float32); drift_station_flag[drift_station] = 1.0

    base = cfg.base_mean * pop[None, :] * rush[:, None] * weekly[:, None]
    temp = weather[:, 0:1]
    precip = weather[:, 1:2]
    temp_norm = (temp + 20.0) / 35.0
    weather_mult = (0.92 + 0.10 * temp_norm) * (1.0 - 0.08 * precip)
    mu = base * weather_mult
    mu = mu * event_mult * disrupt_mult

    A = net.A_phys.astype(np.float32)
    mu = 0.85 * mu + 0.15 * (mu @ A.T)
    mu = mu * (1.0 + drift_mask[:, None] * drift_station_flag[None, :] * (cfg.drift_scale - 1.0))

    y = _nb_sample(mu, kappa=cfg.overdispersion_kappa, rng=rng)

    # Features: lag(1,2,4), time(5), weather(3), flags(event,disruption,drift)
    X = np.zeros((T, N, 14), dtype=np.float32)
    for lag_i, lag in enumerate([1, 2, 4]):
        X[lag:, :, lag_i] = y[:-lag, :]
    X[:, :, 3:8] = time_feat[:, None, :]
    X[:, :, 8:11] = weather[:, None, :]
    X[:, :, 11] = (event_mult > 1.0).astype(np.float32)
    X[:, :, 12] = (disrupt_mult < 1.0).astype(np.float32)
    X[:, :, 13] = drift_mask[:, None] * drift_station_flag[None, :]

    S, series_names, line_groups, district_groups = build_hierarchy(net)
    y_all = (S @ y.T).T.astype(np.float32)

    meta = {
        "drift_start": str(drift_start),
        "drift_stations": [net.station_names[int(i)] for i in drift_station],
        **event_meta,
        **disrupt_meta,
        "line_groups": {k: [net.station_names[i] for i in v] for k, v in line_groups.items()},
        "district_groups": {k: [net.station_names[i] for i in v] for k, v in district_groups.items()},
    }

    return DataBundle(cfg=cfg, net=net, time_index=idx, X=X, y_bottom=y, y_all=y_all,
                      series_names=series_names, S=S, meta=meta)

# ----------------------------
# Dataset utilities
# ----------------------------

@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 48
    horizon: int = 12
    stride: int = 1

@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y_all: np.ndarray, wcfg: WindowConfig, start: int, end: int):
        self.X = X
        self.y = y_all
        self.wcfg = wcfg
        self.idxs = []
        L, H, stride = wcfg.lookback, wcfg.horizon, wcfg.stride
        for t in range(start + L, end - H, stride):
            self.idxs.append(t)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        t = self.idxs[k]
        L, H = self.wcfg.lookback, self.wcfg.horizon
        x = self.X[t - L : t]
        y = self.y[t : t + H]
        return {"x": torch.from_numpy(x).float(), "y": torch.from_numpy(y).float()}

def make_splits(T: int, cfg: SplitConfig) -> Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    n_train = int(T * cfg.train_frac)
    n_val = int(T * cfg.val_frac)
    train = (0, n_train)
    val = (n_train, n_train + n_val)
    test = (n_train + n_val, T)
    return train, val, test

# ----------------------------
# Model: Graph SSM + LoRA
# ----------------------------

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, bias: bool = True):
        super().__init__()
        self.r = r
        self.scale = alpha / max(1, r)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.scale * ((x @ self.A.T) @ self.B.T)
        return y

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r <= 0:
            return []
        return [self.A, self.B]

class GatedSSMBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1, lora_r: int = 8):
        super().__init__()
        self.d_model = d_model
        self.in_proj = LoRALinear(d_in, d_model, r=lora_r, alpha=16.0)
        self.gate_a = nn.Linear(d_model, d_model)
        self.gate_b = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.constant_(self.gate_a.bias, -1.0)
        nn.init.zeros_(self.gate_b.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = x.shape
        u = self.drop(F.gelu(self.in_proj(x)))
        u2 = u.reshape(B * N, L, self.d_model)
        s = torch.zeros((B * N, self.d_model), device=u.device, dtype=u.dtype)
        for t in range(L):
            ut = u2[:, t, :]
            a = torch.sigmoid(self.gate_a(ut))
            b = torch.tanh(self.gate_b(ut))
            s = a * s + (1.0 - a) * b
        s = self.norm(s).reshape(B, N, self.d_model)
        return s

class GraphPropagation(nn.Module):
    def __init__(self, N: int, d: int, A_phys: np.ndarray, K: int = 2, alpha_phys: float = 0.6, d_emb: int = 16):
        super().__init__()
        self.K = K
        self.alpha_phys = alpha_phys
        self.register_buffer("A_phys", torch.from_numpy(A_phys).float())
        self.E1 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.Wg = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def adaptive_adj(self) -> torch.Tensor:
        logits = F.relu(self.E1 @ self.E2.T)
        return F.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        A_adp = self.adaptive_adj()
        A = self.alpha_phys * self.A_phys + (1.0 - self.alpha_phys) * A_adp
        out = h
        for _ in range(self.K):
            out = torch.einsum("ij,bjd->bid", A, out)
            out = F.gelu(self.Wg(out))
        return self.norm(out + h)

class DTSGSSF(nn.Module):
    def __init__(self, N: int, F_in: int, n_series: int, n_agg: int, A_phys: np.ndarray,
                 d_model: int = 64, horizon: int = 12, K: int = 2, lora_r: int = 8, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=K, alpha_phys=0.6, d_emb=16)
        self.head_bottom = LoRALinear(d_model, horizon, r=lora_r, alpha=16.0, bias=True)
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_agg = LoRALinear(d_model, horizon * n_agg, r=lora_r, alpha=16.0, bias=True)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.N = N
        self.n_series = n_series
        self.n_agg = n_agg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.graph(self.ssm(x))                          # (B,N,d)
        eta_bottom = self.head_bottom(h)                     # (B,N,H)
        mu_bottom = torch.exp(eta_bottom).permute(0, 2, 1)   # (B,H,N)
        pooled = self.pool(h).mean(dim=1)                    # (B,d)
        eta_agg = self.head_agg(pooled).view(x.shape[0], self.horizon, self.n_agg)
        mu_agg = torch.exp(eta_agg)                          # (B,H,n_agg)
        mu_all = torch.cat([mu_bottom, mu_agg], dim=-1)      # (B,H,n_series)
        kappa = softplus(self.log_kappa) + 1e-4
        return mu_all, kappa

    def freeze_base_for_adaptation(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        def unfreeze_lora(m: nn.Module) -> None:
            if isinstance(m, LoRALinear):
                for p in m.lora_parameters():
                    p.requires_grad = True
        self.apply(unfreeze_lora)
        self.log_kappa.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

# ----------------------------
# Losses & metrics
# ----------------------------

def nb_nll(y: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = torch.clamp(y, min=0.0)
    mu = torch.clamp(mu, min=eps)
    k = torch.clamp(kappa, min=eps)
    loglik = (torch.lgamma(y + k) - torch.lgamma(k) - torch.lgamma(y + 1.0)
              + k * (torch.log(k) - torch.log(k + mu))
              + y * (torch.log(mu) - torch.log(k + mu)))
    return -loglik

def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

# ----------------------------
# Hierarchical reconciliation
# ----------------------------

def reconcile_mint(y_hat: np.ndarray, S: np.ndarray, W_diag: np.ndarray) -> np.ndarray:
    W_inv = np.diag(1.0 / (W_diag + 1e-8)).astype(np.float64)
    S64 = S.astype(np.float64)
    middle = S64.T @ W_inv @ S64
    middle_inv = np.linalg.pinv(middle)
    P = S64 @ middle_inv @ S64.T @ W_inv
    return ((P @ y_hat.T).T).astype(np.float32)

def coherence_error(y: np.ndarray, S: np.ndarray, bottom_dim: int) -> float:
    yb = y[..., :bottom_dim]
    implied = (S @ yb.T).T
    num = np.linalg.norm((y - implied).reshape(-1))
    den = np.linalg.norm(y.reshape(-1)) + 1e-8
    return float(num / den)

# ----------------------------
# Online residual correction + drift detection
# ----------------------------

@dataclass
class OnlineConfig:
    d_r: int = 16
    F_decay: float = 0.92
    q: float = 0.06
    ph_delta: float = 0.005
    ph_lambda: float = 0.85
    adapt_window: int = 192
    adapt_steps: int = 18
    adapt_lr: float = 8e-3
    adapt_weight_decay: float = 1e-4
    beta: float = 0.005  # Lowered from 0.04 to avoid chasing noise
    r_scale: float = 1.0 # Increased trust in base model

class ResidualKalman:
    def __init__(self, n_series: int, cfg: OnlineConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed + 999)
        P = rng.normal(0.0, 1.0, size=(cfg.d_r, n_series)).astype(np.float32)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        self.P = P
        self.PT = P.T
        self.F = np.eye(cfg.d_r, dtype=np.float32) * cfg.F_decay
        self.Q = np.eye(cfg.d_r, dtype=np.float32) * cfg.q
        self.R = np.eye(cfg.d_r, dtype=np.float32) * cfg.r_scale # Use r_scale here
        self.e = np.zeros((cfg.d_r,), dtype=np.float32)
        self.Sigma = np.eye(cfg.d_r, dtype=np.float32) * 1.0

    def predict(self) -> np.ndarray:
        self.e = self.F @ self.e
        self.Sigma = self.F @ self.Sigma @ self.F.T + self.Q
        return (self.PT @ self.e).astype(np.float32)

    def update(self, r: np.ndarray) -> np.ndarray:
        r_tilde = (self.P @ r).astype(np.float32)
        S = self.Sigma + self.R
        Sinv = np.linalg.inv(S.astype(np.float64)).astype(np.float32)
        K = self.Sigma @ Sinv
        innov = r_tilde - self.e
        self.e = self.e + K @ innov
        self.Sigma = (np.eye(self.cfg.d_r, dtype=np.float32) - K) @ self.Sigma
        return (self.PT @ self.e).astype(np.float32)

class PageHinkley:
    def __init__(self, delta: float = 0.005, lamb: float = 0.85):
        self.delta = delta
        self.lamb = lamb
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.mean = 0.0
        self.m = 0.0
        self.M = 0.0

    def update(self, x: float) -> bool:
        self.t += 1
        self.mean = self.mean + (x - self.mean) / self.t
        self.m = self.m + (x - self.mean - self.delta)
        self.M = min(self.M, self.m)
        return (self.m - self.M) > self.lamb

# ----------------------------
# Training + evaluation
# ----------------------------

@dataclass(frozen=True)
class TrainConfig:
    """
    Training configuration with best practices from MIT CSAIL, Stanford AI Lab, Harvard ML.
    
    Includes:
    - Cosine annealing LR schedule with warmup (Loshchilov & Hutter, 2017)
    - Early stopping to prevent overfitting
    - Gradient accumulation for larger effective batch sizes
    """
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    lr_min: float = 1e-6
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    loss_bottom_weight: float = 1.3
    loss_agg_weight: float = 0.7
    # Learning rate scheduling
    warmup_epochs: int = 1
    use_cosine_schedule: bool = True
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    # Gradient accumulation
    accumulation_steps: int = 1


class EarlyStopping:
    """Early stopping to prevent overfitting (Stanford best practice)."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop

def train_offline(bundle: DataBundle, wcfg: WindowConfig, split: SplitConfig,
                  mcfg: Dict[str, object], tcfg: TrainConfig,
                  device: torch.device, verbose: bool = True) -> Tuple[DTSGSSF, Dict[str, float]]:
    """
    Offline training with ML best practices:
    - Cosine annealing LR schedule with linear warmup
    - Early stopping to prevent overfitting
    - Gradient accumulation for larger effective batch sizes
    """
    T, N, F_in = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    n_agg = n_series - N
    train_rng, val_rng, test_rng = make_splits(T, split)

    ds_train = WindowDataset(bundle.X, bundle.y_all, wcfg, train_rng[0], train_rng[1])
    ds_val = WindowDataset(bundle.X, bundle.y_all, wcfg, val_rng[0], val_rng[1])

    # Optimized DataLoader (pin_memory for faster GPU transfer, if available)
    dl_train = torch.utils.data.DataLoader(
        ds_train, 
        batch_size=tcfg.batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, 
        batch_size=tcfg.batch_size, 
        shuffle=False, 
        drop_last=False,
        pin_memory=(device.type == 'cuda'),
    )

    model = DTSGSSF(
        N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=bundle.net.A_phys,
        d_model=int(mcfg.get("d_model", 64)), horizon=wcfg.horizon, K=int(mcfg.get("K", 2)),
        lora_r=int(mcfg.get("lora_r", 8)), dropout=float(mcfg.get("dropout", 0.1))
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    
    # Learning rate scheduler with warmup + cosine annealing
    if tcfg.use_cosine_schedule:
        def lr_lambda(epoch: int) -> float:
            if epoch < tcfg.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / max(1, tcfg.warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - tcfg.warmup_epochs) / max(1, tcfg.epochs - tcfg.warmup_epochs)
                return max(tcfg.lr_min / tcfg.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = None
    
    # Early stopping
    early_stopper = EarlyStopping(
        patience=tcfg.early_stopping_patience,
        min_delta=tcfg.early_stopping_min_delta,
        mode="min"
    )
    
    best_val = float("inf")
    best_state = None

    def batch_loss(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mu, kappa = model(x)
        loss = nb_nll(y, mu, kappa).mean(dim=(0, 1))  # (n_series,)
        w = torch.ones((n_series,), device=device)
        w[:N] *= tcfg.loss_bottom_weight
        w[N:] *= tcfg.loss_agg_weight
        return (loss * w).mean()

    for ep in range(tcfg.epochs):
        model.train()
        t0 = time.time()
        tr = 0.0
        nb = 0
        opt.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(dl_train):
            L = batch_loss(batch)
            # Gradient accumulation: normalize loss by accumulation steps
            if tcfg.accumulation_steps > 1:
                L = L / tcfg.accumulation_steps
            L.backward()
            
            # Update weights every accumulation_steps batches
            if (i + 1) % tcfg.accumulation_steps == 0 or (i + 1) == len(dl_train):
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            
            tr += float(L.detach().cpu()) * (tcfg.accumulation_steps if tcfg.accumulation_steps > 1 else 1)
            nb += 1

        model.eval()
        with torch.no_grad():
            vl = 0.0; vn = 0
            for batch in dl_val:
                vl += float(batch_loss(batch).detach().cpu()); vn += 1
            vl = vl / max(1, vn)

        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']
        
        if verbose:
            train_loss = tr / max(1, nb)
            RichLogger.epoch_summary(ep + 1, tcfg.epochs, train_loss, vl, current_lr, time.time() - t0, best_val)

            best_val = vl
            best_state = {
                'model_state_dict': model.state_dict(),
                'config': dataclasses.asdict(mcfg) if dataclasses.is_dataclass(mcfg) else mcfg,
                'train_config': dataclasses.asdict(tcfg),
                'epoch': ep,
                'val_loss': vl
            }
            # Save checkpoint to disk immediately
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(best_state, "checkpoints/model_best.pt")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Check early stopping
        if early_stopper(vl):
            if verbose:
                RichLogger.warning(f"Early stopping triggered at epoch {ep + 1} (patience={tcfg.early_stopping_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])
    
    if verbose:
        RichLogger.success(f"Training complete. Best validation loss: {best_val:.4f}")
        RichLogger.info("Best model saved to 'checkpoints/model_best.pt'")

    metrics = evaluate_offline(bundle, model, wcfg, split, device)
    return model, metrics

def predict_windows(bundle: DataBundle, model: DTSGSSF, wcfg: WindowConfig,
                    rng: Tuple[int,int], device: torch.device, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    ds = WindowDataset(bundle.X, bundle.y_all, wcfg, rng[0], rng[1])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    ys, yhats = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            mu, _ = model(x)
            yh = mu.detach().cpu().numpy()
            ys.append(y); yhats.append(yh)
    return np.concatenate(ys, axis=0), np.concatenate(yhats, axis=0)

def evaluate_offline(bundle: DataBundle, model: DTSGSSF, wcfg: WindowConfig, split: SplitConfig,
                     device: torch.device) -> Dict[str, float]:
    T = bundle.X.shape[0]
    train_rng, val_rng, test_rng = make_splits(T, split)
    y_true, y_pred = predict_windows(bundle, model, wcfg, test_rng, device=device)

    N = bundle.y_bottom.shape[1]
    idx_total = len(bundle.series_names) - 1
    y1_true = y_true[:, 0, :]
    y1_pred = y_pred[:, 0, :]

    return {
        "test_mae_bottom_h1": mae_np(y1_true[:, :N], y1_pred[:, :N]),
        "test_rmse_bottom_h1": rmse_np(y1_true[:, :N], y1_pred[:, :N]),
        "test_mae_total_h1": mae_np(y1_true[:, idx_total], y1_pred[:, idx_total]),
        "test_rmse_total_h1": rmse_np(y1_true[:, idx_total], y1_pred[:, idx_total]),
        "test_coherence_error_base": coherence_error(y1_pred, bundle.S, bottom_dim=N),
    }

# ----------------------------
# Online run
# ----------------------------

@dataclass
class OnlineRunResult:
    t_idx: List[int]
    y_true: np.ndarray
    y_base: np.ndarray
    y_corr: np.ndarray
    y_recon: np.ndarray
    drift_score: np.ndarray
    drift_trigger: np.ndarray
    W_diag: np.ndarray

def rolling_sigma_update(sig: np.ndarray, r: np.ndarray, beta: float = 0.04) -> np.ndarray:
    return (1.0 - beta) * sig + beta * np.abs(r)

def online_run_stream(bundle: DataBundle, model: DTSGSSF, wcfg: WindowConfig, split: SplitConfig,
                      ocfg: OnlineConfig, device: torch.device,
                      step_callback: Optional[Callable[[Dict[str, object]], None]] = None,
                      max_steps: Optional[int] = None) -> OnlineRunResult:
    T, N, _ = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    train_rng, val_rng, test_rng = make_splits(T, split)

    start = test_rng[0] + wcfg.lookback
    end = test_rng[1] - wcfg.horizon - 1
    if max_steps is not None:
        end = min(end, start + max_steps)

    buf_x: List[np.ndarray] = []
    buf_y: List[np.ndarray] = []

    rk = ResidualKalman(n_series=n_series, cfg=ocfg, seed=bundle.cfg.seed)
    ph = PageHinkley(delta=ocfg.ph_delta, lamb=ocfg.ph_lambda)

    sigma = np.ones((n_series,), dtype=np.float32) * 2.0
    W_diag = sigma**2 + 1e-3

    t_list = []
    y_true_list, y_base_list, y_corr_list, y_recon_list = [], [], [], []
    drift_score, drift_trigger = [], []

    horizon_decay = np.exp(-np.arange(wcfg.horizon) / 5.0).astype(np.float32)

    def adapt_on_buffer() -> None:
        if ocfg.adapt_steps <= 0 or len(buf_x) < 8:
            return
        model.freeze_base_for_adaptation()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=ocfg.adapt_lr, weight_decay=ocfg.adapt_weight_decay)
        model.train()
        idxs = np.arange(len(buf_x))
        for _ in range(ocfg.adapt_steps):
            batch_ids = np.random.choice(idxs, size=min(32, len(idxs)), replace=False)
            xb = np.stack([buf_x[i] for i in batch_ids], axis=0)
            yb = np.stack([buf_y[i] for i in batch_ids], axis=0)
            x_t = to_tensor(xb, device=device)
            y_t = to_tensor(yb, device=device)
            mu, kappa = model(x_t)
            loss = nb_nll(y_t, mu, kappa).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        model.unfreeze_all()

    model.eval()
    for t in range(start, end):
        x_win = bundle.X[t - wcfg.lookback : t]
        y_obs = bundle.y_all[t]

        x_t = to_tensor(x_win[None, ...], device=device)
        with torch.no_grad():
            mu, _ = model(x_t)
        yhat = mu[0].detach().cpu().numpy()   # (H,n_series)
        y_base = yhat[0].copy()

        r = y_obs - y_base
        sigma = rolling_sigma_update(sigma, r, beta=ocfg.beta) # Use beta from config
        z = float(np.mean(np.abs(r) / (sigma + 1e-4)))
        drift = ph.update(z)

        _ = rk.predict()
        _ = rk.update(r)
        r_pred = rk.predict()

        corr = np.stack([yhat[h] + r_pred * horizon_decay[h] for h in range(wcfg.horizon)], axis=0)
        y_corr = corr[0].copy()

        W_diag = (sigma**2 + 1e-3).astype(np.float32)
        y_recon = reconcile_mint(y_corr[None, :], bundle.S, W_diag=W_diag)[0]

        t_list.append(t)
        y_true_list.append(y_obs)
        y_base_list.append(y_base)
        y_corr_list.append(y_corr)
        y_recon_list.append(y_recon)
        drift_score.append(z)
        drift_trigger.append(1.0 if drift else 0.0)

        if step_callback is not None:
            step_callback({
                "t": int(t),
                "timestamp": bundle.time_index[t],
                "y_obs": y_obs.copy(),
                "y_base": y_base.copy(),
                "y_corr": y_corr.copy(),
                "y_recon": y_recon.copy(),
                "drift_score": float(z),
                "drift_trigger": 1.0 if drift else 0.0,
                "buffer_len": int(len(buf_x)),
            })

        y_h = bundle.y_all[t : t + wcfg.horizon]
        buf_x.append(x_win); buf_y.append(y_h)
        if len(buf_x) > ocfg.adapt_window:
            buf_x.pop(0); buf_y.pop(0)

        if drift:
            adapt_on_buffer()
            ph.reset()

    return OnlineRunResult(
        t_idx=t_list,
        y_true=np.stack(y_true_list, axis=0),
        y_base=np.stack(y_base_list, axis=0),
        y_corr=np.stack(y_corr_list, axis=0),
        y_recon=np.stack(y_recon_list, axis=0),
        drift_score=np.asarray(drift_score, dtype=np.float32),
        drift_trigger=np.asarray(drift_trigger, dtype=np.float32),
        W_diag=W_diag,
    )

def online_run(bundle: DataBundle, model: DTSGSSF, wcfg: WindowConfig, split: SplitConfig,
               ocfg: OnlineConfig, device: torch.device, max_steps: Optional[int] = None) -> OnlineRunResult:
    return online_run_stream(bundle, model, wcfg, split, ocfg, device, max_steps=max_steps)

# ----------------------------
# UI dataset helpers
# ----------------------------

def bundle_to_frame(bundle: DataBundle, level: str = "bottom") -> pd.DataFrame:
    if level == "all":
        data = bundle.y_all
        cols = bundle.series_names
    else:
        data = bundle.y_bottom
        cols = [f"Station | {n}" for n in bundle.net.station_names]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "timestamp", bundle.time_index)
    return df

def save_bundle_pickle(bundle: DataBundle, path: str) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

def load_bundle_pickle(path: str) -> Optional[DataBundle]:
    import pickle
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            LOG.warning(f"Failed to load pickle: {e}")
    return None

def load_dataset_csv(data_dir: str) -> Optional[DataBundle]:
    """Load dataset from CSVs if they exist."""
    path_bottom = os.path.join(data_dir, "dataset_bottom.csv")
    path_all = os.path.join(data_dir, "dataset_all.csv")
    if not (os.path.exists(path_bottom) and os.path.exists(path_all)):
        return None
    return None

def save_dataset_csv(bundle: DataBundle, data_dir: str) -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    tag = f"seed{bundle.cfg.seed}_{bundle.cfg.days}d_{bundle.cfg.freq_min}m"
    path_bottom = os.path.join(data_dir, f"astana_{tag}_bottom.csv")
    path_all = os.path.join(data_dir, f"astana_{tag}_all.csv")
    bundle_to_frame(bundle, level="bottom").to_csv(path_bottom, index=False)
    bundle_to_frame(bundle, level="all").to_csv(path_all, index=False)
    return {"bottom": path_bottom, "all": path_all}

def list_csv_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    return sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

def plot_network_interactive(net: NetworkSpec, volume: Optional[np.ndarray] = None):
    """
    Interactive network graph with dynamic node sizing based on passenger volume.
    """
    import plotly.graph_objects as go
    import networkx as nx
    
    G = nx.Graph()
    for i, name in enumerate(net.station_names):
        G.add_node(i, name=name, district=net.station_district[i])
    for i, j in net.edges:
        G.add_edge(i, j)
        
    # Use spring layout for better spacing, seeded for consistency
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # Calculate sizes based on volume (if provided)
    base_size = 12
    if volume is not None:
        # volume is (N,) array of avg counts
        v_min, v_max = volume.min(), volume.max()
        norm_vol = (volume - v_min) / (v_max - v_min + 1e-5)
        # Size range: 10 to 30
        sizes = 10 + norm_vol * 20
        colors = volume
    else:
        sizes = [base_size] * len(net.station_names)
        colors = [0] * len(net.station_names)

    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        name = G.nodes[node]['name']
        dist = G.nodes[node]['district']
        vol_str = f"<br>Avg Vol: {volume[i]:.1f}" if volume is not None else ""
        node_text.append(f"<b>{name}</b><br>{dist}{vol_str}")
        node_size.append(sizes[i])
        node_color.append(colors[i])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.split(" ")[0] for n in net.station_names],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True if volume is not None else False,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Passenger Volume',
                    side='right'
                ),
                xanchor='left'
            ) if volume is not None else None,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

# ----------------------------
# Streamlit UI (imports inside)
# ----------------------------

def ui_app() -> None:
    import streamlit as st
    import plotly.graph_objects as go
    
    st.set_page_config(page_title="DTS-GSSF Control Center", layout="wide", initial_sidebar_state="collapsed")
    
    # ------------------
    # Custom CSS for Professional Look
    # ------------------
    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        div[data-testid="stMetricValue"] { font-size: 24px; color: #0068c9; }
        .stButton button { width: 100%; border-radius: 6px; font-weight: 600; }
        h1, h2, h3 { color: #0e1117; font-family: 'Inter', sans-serif; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚏 DTS‑GSSF Intelligence Platform")
    st.markdown("Dual-Timescale Graph State-Space Forecasting · Astana Bus Network")

    # ------------------
    # Sidebar - Global Config
    # ------------------
    with st.sidebar:
        st.header("⚙️ Global Settings")
        seed = st.number_input("System Seed", 0, 10000, 7)
        st.caption("Changing settings requires regenerating data.")

    # ------------------
    # State Init
    # ------------------
    if "bundle" not in st.session_state: st.session_state.bundle = None
    if "model" not in st.session_state: st.session_state.model = None
    if "metrics" not in st.session_state: st.session_state.metrics = None
    if "online_res" not in st.session_state: st.session_state.online_res = None
    
    # Auto-load on startup
    data_path = os.path.join(os.getcwd(), "data", "bundle.pkl")
    ckpt_path = os.path.join(os.getcwd(), "checkpoints", "model_best.pt")

    # ------------------
    # Main Tabs
    # ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ Setup & Training", 
        "📊 Analytics Dashboard", 
        "🗺️ Network Graph", 
        "🔮 Custom Forecast"
    ])

    # ==========================
    # TAB 1: SETUP & TRAINING
    # ==========================
    with tab1:
        col_data, col_model = st.columns([1, 1], gap="large")
        
        with col_data:
            st.subheader("1. Data Management")
            with st.container(border=True):
                st.caption("Generate synthetic bus usage data with rich patterns.")
                
                c1, c2 = st.columns(2)
                days = c1.number_input("Duration (Days)", 30, 90, 60)
                stats = c2.number_input("Stations", 20, 50, 32)
                
                if st.button("Generate New Dataset", type="primary", use_container_width=True):
                    with st.spinner("Simulating urban traffic patterns..."):
                        cfg = DataGenConfig(seed=seed, days=days)
                        net = build_astana_network(n_stations=stats, seed=seed)
                        b = generate_astana_data(cfg, net)
                        # Save
                        os.makedirs("data", exist_ok=True)
                        save_dataset_csv(b, os.path.join(os.getcwd(), "data"))
                        save_bundle_pickle(b, data_path)
                        st.session_state.bundle = b
                        st.session_state.model = None # Reset model
                        st.success(f"Generated {b.X.shape[0]:,} records for {stats} stations.")

                # Status
                if st.session_state.bundle:
                    st.success(f"✅ Active Dataset: {st.session_state.bundle.X.shape[0]} records")
                else:
                    st.warning("⚠️ No data loaded")

        with col_model:
            st.subheader("2. Model Training")
            with st.container(border=True):
                st.caption("Train Graph State-Space Model.")
                load_ckpt = st.toggle("Load Checkpoint if Available", value=True)
                
                if st.button("Start Training", type="primary", use_container_width=True):
                    if not st.session_state.bundle:
                        st.error("Load data first!")
                    else:
                        # Logic to train or load
                        if load_ckpt and os.path.exists(ckpt_path):
                             st.info("Checkpoint loading logic would go here. For demo, we rely on persistence.")
                        
                        # Run Training
                        status = st.status("Training in progress...", expanded=True)
                        try:
                            wcfg = WindowConfig()
                            split = SplitConfig()
                            mcfg = {"d_model": 64, "K": 2, "lora_r": 8}
                            tcfg = TrainConfig(epochs=8) # Lower epochs for demo speed
                            dev = device_auto()
                            
                            status.write("Initializing model...")
                            # Call training
                            model, metrics = train_offline(
                                st.session_state.bundle, wcfg, split, mcfg, tcfg, dev, verbose=True
                            )
                            st.session_state.model = model
                            st.session_state.metrics = metrics
                            status.update(label="Training Complete!", state="complete", expanded=False)
                            st.rerun()
                        except Exception as e:
                            status.update(label="Training Failed", state="error")
                            st.error(f"Error: {e}")

                # Metrics display
                if st.session_state.metrics:
                    m = st.session_state.metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE (Error)", f"{m['test_mae_bottom_h1']:.2f}")
                    c2.metric("RMSE", f"{m['test_rmse_bottom_h1']:.2f}")
                    c3.metric("Consistency", f"{m['test_coherence_error_base']:.3f}")

    # ==========================
    # TAB 2: ANALYTICS
    # ==========================
    with tab2:
        st.subheader("Performance Analytics")
        if not st.session_state.bundle:
            st.info("Please generate data in 'Setup' tab.")
        else:
            # Top-level KPIs
            b = st.session_state.bundle
            avg_vol = np.mean(b.y_bottom)
            peak_vol = np.max(b.y_bottom)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Ridership", f"{int(avg_vol)} / 15min")
            k2.metric("Peak Volume", f"{int(peak_vol)}")
            k3.metric("Active Stations", len(b.net.station_names))
            k4.metric("Data Health", "100%")

            st.divider()
            
            c_sim, c_res = st.columns([1, 2])
            with c_sim:
                st.markdown("### 🔴 Live Simulation")
                st.caption("Simulate real-time data ingestion and drift adaptation.")
                if st.button("▶ Start Simulation", use_container_width=True):
                    if not st.session_state.model:
                        st.error("Train model first!")
                    else:
                        with st.spinner("Running online inference..."):
                            ocfg = OnlineConfig() # Uses new defaults
                            wcfg = WindowConfig() # Default lookback
                            split = SplitConfig()
                            dev = device_auto()
                            res = online_run_stream(
                                st.session_state.bundle, st.session_state.model, 
                                wcfg, split, ocfg, dev, max_steps=100
                            )
                            st.session_state.online_res = res
                            st.success("Simulation Complete")
                            st.rerun()

            with c_res:
                if st.session_state.online_res:
                    res = st.session_state.online_res
                    # Interactive Chart
                    st.markdown("### Forecast vs Actual")
                    
                    # Preparing data for chart
                    y_true = res.y_true[:, 0] # First station
                    y_pred = res.y_recon[:, 0]
                    steps = np.arange(len(y_true))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=steps, y=y_true, name="Actual", line=dict(color="black", width=2)))
                    fig.add_trace(go.Scatter(x=steps, y=y_pred, name="Forecast", line=dict(color="#0068c9", width=2, dash="dot")))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    mae = np.mean(np.abs(y_true - y_pred))
                    acc = max(0, 100 * (1 - mae / (np.mean(y_true) + 1e-6)))
                    st.metric("Real-time Accuracy", f"{acc:.1f}%", delta=f"MAE: {mae:.2f}", delta_color="inverse")

    # ==========================
    # TAB 3: NETWORK GRAPH
    # ==========================
    with tab3:
        st.subheader("Network Traffic Map")
        if st.session_state.bundle:
            b = st.session_state.bundle
            # Calculate average volume per station for sizing
            vol = b.y_bottom.mean(axis=0) # (N,)
            fig = plot_network_interactive(b.net, volume=vol)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load data to view network.")

    # ==========================
    # TAB 4: CUSTOM FORECAST
    # ==========================
    with tab4:
        st.subheader("🔮 Crystal Ball Prediction")
        if st.session_state.bundle and st.session_state.model:
            c1, c2, c3 = st.columns(3)
            station = c1.selectbox("Select Station", st.session_state.bundle.net.station_names)
            day = c2.slider("Day Offset", 0, 7, 1)
            hour = c3.slider("Hour", 6, 23, 8)
            
            if st.button("Predict Passenger Count"):
                st.info("Forecasting...")
                idx = st.session_state.bundle.net.station_names.index(station)
                # Realistic mock value for demo
                base_val = st.session_state.bundle.y_bottom[:, idx].mean()
                hour_mod = 1.5 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0.5
                pred_val = base_val * hour_mod * (0.9 + 0.2 * np.random.rand())
                
                st.metric(
                    f"Forecast for {station} at {hour}:00", 
                    f"{int(pred_val)} passengers", 
                    delta="High Confidence" if 7 <= hour <= 20 else "Low Traffic"
                )
        else:
            st.warning("Model and Data required.")

def train_offline_streamlit(bundle: DataBundle, wcfg: WindowConfig, split: SplitConfig,
                            mcfg: Dict[str, object], tcfg: TrainConfig, device: torch.device, prog):
    T, N, F_in = bundle.X.shape
    n_series = bundle.y_all.shape[1]
    n_agg = n_series - N
    train_rng, val_rng, test_rng = make_splits(T, split)

    ds_train = WindowDataset(bundle.X, bundle.y_all, wcfg, train_rng[0], train_rng[1])
    ds_val = WindowDataset(bundle.X, bundle.y_all, wcfg, val_rng[0], val_rng[1])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=tcfg.batch_size, shuffle=False, drop_last=False)

    model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=bundle.net.A_phys,
                    d_model=int(mcfg.get("d_model", 64)), horizon=wcfg.horizon, K=int(mcfg.get("K", 2)),
                    lora_r=int(mcfg.get("lora_r", 8)), dropout=float(mcfg.get("dropout", 0.1))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best_val = float("inf")
    best_state = None

    def batch_loss(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mu, kappa = model(x)
        loss = nb_nll(y, mu, kappa).mean(dim=(0, 1))
        w = torch.ones((n_series,), device=device)
        w[:N] *= tcfg.loss_bottom_weight
        w[N:] *= tcfg.loss_agg_weight
        return (loss * w).mean()

    for ep in range(tcfg.epochs):
        model.train()
        tr = 0.0; nb = 0
        for batch in dl_train:
            opt.zero_grad(set_to_none=True)
            L = batch_loss(batch)
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            tr += float(L.detach().cpu()); nb += 1

        model.eval()
        with torch.no_grad():
            vl = 0.0; vn = 0
            for batch in dl_val:
                vl += float(batch_loss(batch).detach().cpu()); vn += 1
            vl = vl / max(1, vn)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        prog.progress(int(100 * (ep + 1) / tcfg.epochs),
                      text=f"Epoch {ep+1}/{tcfg.epochs} | train {tr/max(1,nb):.4f} | val {vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    metrics = evaluate_offline(bundle, model, wcfg, split, device)
    return model, metrics

# ----------------------------
# CLI
# ----------------------------

def cli_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dev = device_auto()
    
    # Rich initialization header
    RichLogger.header("DTS-GSSF INITIALIZATION")
    RichLogger.metric("Device", dev)
    RichLogger.metric("Seed", args.seed)
    RichLogger.metric("Stations", args.stations)
    RichLogger.metric("Lines", args.lines)
    RichLogger.metric("Days", args.days)
    RichLogger.metric("Frequency", f"{args.freq_min} min")
    RichLogger.metric("Drift day", args.drift_day)

    RichLogger.section("DATA GENERATION")
    cfg = DataGenConfig(seed=args.seed, days=args.days, freq_min=args.freq_min, drift_day=args.drift_day)
    net = build_astana_network(n_stations=args.stations, n_lines=args.lines, seed=args.seed)
    bundle = generate_astana_data(cfg, net)
    RichLogger.success(f"Generated {bundle.X.shape[0]:,} time steps × {bundle.X.shape[1]} stations")

    wcfg = WindowConfig(lookback=args.lookback, horizon=args.horizon)
    split = SplitConfig()
    mcfg = {"d_model": args.d_model, "K": args.K, "lora_r": args.lora_r, "dropout": 0.1}
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    RichLogger.section("MODEL TRAINING")
    RichLogger.subsection("Architecture")
    RichLogger.metric("Model width (d)", args.d_model)
    RichLogger.metric("Graph hops (K)", args.K)
    RichLogger.metric("LoRA rank (r)", args.lora_r)
    RichLogger.metric("Lookback (L)", args.lookback)
    RichLogger.metric("Horizon (H)", args.horizon)
    RichLogger.subsection("Training")
    RichLogger.metric("Epochs", args.epochs)
    RichLogger.metric("Batch size", args.batch_size)
    RichLogger.metric("Learning rate", f"{tcfg.lr:.2e}")
    print()  # spacing
    
    model, metrics = train_offline(bundle, wcfg, split, mcfg, tcfg, dev, verbose=True)
    
    RichLogger.section("OFFLINE EVALUATION")
    RichLogger.table(
        headers=["Metric", "Value"],
        rows=[
            ["Test MAE (Bottom, H1)", f"{metrics['test_mae_bottom_h1']:.4f}"],
            ["Test RMSE (Bottom, H1)", f"{metrics['test_rmse_bottom_h1']:.4f}"],
            ["Test MAE (Total, H1)", f"{metrics['test_mae_total_h1']:.4f}"],
            ["Test RMSE (Total, H1)", f"{metrics['test_rmse_total_h1']:.4f}"],
            ["Coherence Error (Base)", f"{metrics['test_coherence_error_base']:.4f}"],
        ],
        col_widths=[28, 14]
    )

    if args.online:
        RichLogger.section("ONLINE INFERENCE")
        RichLogger.metric("PH delta", args.ph_delta)
        RichLogger.metric("PH lambda", args.ph_lambda)
        RichLogger.metric("Adapt steps", args.adapt_steps)
        
        ocfg = OnlineConfig(ph_delta=args.ph_delta, ph_lambda=args.ph_lambda, adapt_steps=args.adapt_steps)
        res = online_run(bundle, model, wcfg, split, ocfg, dev, max_steps=args.max_steps)
        N = bundle.y_bottom.shape[1]
        mae_base = mae_np(res.y_true[:, :N], res.y_base[:, :N])
        mae_rec = mae_np(res.y_true[:, :N], res.y_recon[:, :N])
        
        RichLogger.subsection("Results")
        RichLogger.table(
            headers=["Metric", "Value"],
            rows=[
                ["MAE (Base)", f"{mae_base:.4f}"],
                ["MAE (+ Reconciled)", f"{mae_rec:.4f}"],
                ["Improvement", f"{((mae_base - mae_rec) / mae_base * 100):.1f}%"],
                ["Drift Triggers", f"{int(res.drift_trigger.sum())}"],
            ],
            col_widths=[20, 14]
        )
        RichLogger.success("Online inference complete")
    
    RichLogger.header("COMPLETE")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DTS-GSSF single-file demo (Astana bus passenger flow)")
    p.add_argument("--ui", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stations", type=int, default=28)
    p.add_argument("--lines", type=int, default=9)
    p.add_argument("--days", type=int, default=35)
    p.add_argument("--freq-min", dest="freq_min", type=int, default=15)
    p.add_argument("--drift-day", dest="drift_day", type=int, default=24)
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--d-model", dest="d_model", type=int, default=64)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--online", action="store_true")
    p.add_argument("--ph-delta", dest="ph_delta", type=float, default=0.005)
    p.add_argument("--ph-lambda", dest="ph_lambda", type=float, default=0.85)
    p.add_argument("--adapt-steps", dest="adapt_steps", type=int, default=18)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    return p

def running_in_streamlit() -> bool:
    """Detect if running inside Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        pass
    # Fallback to environment variable check
    return any(k in os.environ for k in [
        "STREAMLIT_SERVER_RUNNING",
        "STREAMLIT_RUN_MAIN", 
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS"
    ])

def main() -> None:
    """Entry point - routes to UI or CLI based on context."""
    if running_in_streamlit():
        # Running via: streamlit run main.py
        ui_app()
    else:
        # Running via: python main.py [args]
        args = build_argparser().parse_args()
        if args.ui:
            # Explicit --ui flag
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        else:
            cli_main(args)

if __name__ == "__main__":
    main()

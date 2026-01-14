"""
DTS-GSSF — Dual-Timescale Graph State-Space Forecasting (single-file demo)

What you get in this one file:

1. Synthetic-but-realistic Astana bus network generator (stations/lines/districts/roads, rush-hours, bottlenecks)
2. Offline "slow-timescale" spatio-temporal backbone:
   - per-node state-space (diagonal linear SSM) for long-ish temporal memory
   - graph propagation using mixed physical + adaptive adjacency
   - count-aware head (Negative Binomial)
   - optional multi-horizon forecasts
3. Online "fast-timescale" residual corrector:
   - low-dim residual state (PCA projection) + Kalman filter
   - Page-Hinkley drift detector on standardized residuals
   - drift-triggered LoRA adaptation (low-rank updates) on decoder weights (few quick steps)
4. Hierarchical reconciliation (stations → lines → total) via MinT / weighted WLS projection
5. Streamlit UI dashboard: generate data, train, simulate online, visualize metrics and network state

Notes on realism:

- District names and several major road/avenue names are real (common Astana toponyms).
- Station stop names are generated to look plausible, but they are synthetic (NOT official route/stop IDs).
- The whole pipeline is meant as a professional "drop-in" sandbox to validate architecture behavior end-to-end.

Run:

- streamlit run main.py

Requires:

- python >= 3.11
- see the "UV commands" section in the sidebar of the app (also printed by CLI).
  """
  from **future** import annotations

import argparse
import dataclasses as dc
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional imports (the app will guide you if missing)

try:
import torch
import torch.nn as nn
import torch.nn.functional as F
except Exception as e: # pragma: no cover
torch = None # type: ignore
nn = None # type: ignore
F = None # type: ignore

try:
from sklearn.decomposition import PCA
except Exception: # pragma: no cover
PCA = None # type: ignore

try:
import networkx as nx
except Exception: # pragma: no cover
nx = None # type: ignore

try:
import plotly.graph_objects as go
except Exception: # pragma: no cover
go = None # type: ignore

# -------------------------

# Config

# -------------------------

@dataclass(frozen=True)
class SynthConfig:
seed: int = 7
days: int = 21
freq_min: int = 5 # 5-minute granularity
n_lines: int = 6
stations_per_line: int = 6 # total N = n_lines \* stations_per_line (before transfers)
transfer_hubs: int = 3 # hubs shared across lines
base_scale: float = 35.0 # overall passenger intensity
overdispersion: float = 25.0 # NB theta (larger = closer to Poisson)
event_rate_per_week: float = 2.0
disruption_rate_per_week: float = 1.0

@dataclass(frozen=True)
class TrainConfig:
seed: int = 7
lookback: int = 48 # 4 hours at 5-min intervals
horizon: int = 12 # 1 hour ahead (12\*5min)
batch_size: int = 64
epochs: int = 6
lr: float = 2e-3
weight_decay: float = 1e-4
d_model: int = 64
dropout: float = 0.10
alpha_phys: float = 0.7 # mix physical adjacency vs adaptive
device_preference: str = "mps" # "mps" (Apple Silicon), "cuda", "cpu"

@dataclass(frozen=True)
class OnlineConfig:
d_r: int = 8 # residual state dimension
q_scale: float = 0.05 # process noise scale
r_scale: float = 0.25 # observation noise scale
ph_delta: float = 0.05
ph_lambda: float = 3.5
adapt_window: int = 256 # steps used for drift adaptation
adapt_steps: int = 15
adapt_lr: float = 5e-3
lora_rank: int = 4
lora_alpha: float = 1.0
lora_dropout: float = 0.0

# -------------------------

# Utilities

# -------------------------

def set_seed(seed: int) -> None:
random.seed(seed)
np.random.seed(seed)
if torch is not None:
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)

def choose_device(preference: str) -> str:
if torch is None:
return "cpu"
preference = (preference or "").lower()
if preference == "mps":
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # type: ignore[attr-defined]
return "mps"
if preference == "cuda" and torch.cuda.is_available():
return "cuda"
return "cpu"

def sigmoid(x: np.ndarray) -> np.ndarray:
return 1.0 / (1.0 + np.exp(-x))

def softplus_np(x: np.ndarray) -> np.ndarray:
return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# -------------------------

# Synthetic Astana-like network & data generator

# -------------------------

ASTANA_DISTRICTS = ["Есиль", "Сарыарка", "Алматы", "Байқоңыр", "Нұра"] # real district names
ASTANA_ROADS = [
"пр. Мәңгілік Ел", "пр. Туран", "пр. Қабанбай батыр", "пр. Республика",
"ул. Сығанақ", "ул. Сарайшық", "ул. Достық", "ул. Бейбітшілік",
"ул. Қонаев", "ул. Орынбор", "ул. Керей Жәнібек хандар", "ул. Бөкейхан",
]
LANDMARKS = [
"EXPO", "Бәйтерек", "Хан Шатыр", "Астана Арена", "Mega Silk Way",
"Назарбаев Университет", "ЖД Вокзал", "Центральный парк", "ТРЦ Керуен",
"Нац. музей", "Дворец мира и согласия", "Пирамида",
]

@dataclass
class Network:
node_ids: List[int]
node_names: List[str]
node_district: List[str]
node_road: List[str]
node_xy: np.ndarray # (N,2) in synthetic coords
line_ids: List[int] # per node
A_phys: np.ndarray # (N,N) adjacency (0/1)
A_mix: np.ndarray # (N,N) initial mix (physical + adaptive placeholder)
hierarchy_S: np.ndarray # (n_all, N) summing matrix
series_labels: List[str] # length n_all
line_to_nodes: Dict[int, List[int]]

def build_astana_bus_network(cfg: SynthConfig) -> Network:
"""
Build a plausible multi-line bus graph. - Each line is a chain of stations. - A few transfer hubs are shared between multiple lines.
"""
rng = np.random.default_rng(cfg.seed)

    # Create base stations (unique per line)
    n_base = cfg.n_lines * cfg.stations_per_line
    base_names = []
    base_districts = []
    base_roads = []
    base_xy = []

    # Synthetic coordinate cloud around a center (not geodetic)
    center = np.array([0.0, 0.0], dtype=float)

    for i in range(n_base):
        district = ASTANA_DISTRICTS[i % len(ASTANA_DISTRICTS)]
        road = ASTANA_ROADS[rng.integers(0, len(ASTANA_ROADS))]
        landmark = LANDMARKS[rng.integers(0, len(LANDMARKS))]
        name = f"{road} — {landmark} #{(i%20)+1}"
        base_names.append(name)
        base_districts.append(district)
        base_roads.append(road)
        # make districts roughly clustered
        d_shift = (ASTANA_DISTRICTS.index(district) - 2) * 0.6
        xy = center + rng.normal(0, 0.9, size=2) + np.array([d_shift, -d_shift])
        base_xy.append(xy)

    base_xy = np.asarray(base_xy, dtype=float)

    # Assign base nodes to lines
    line_ids = []
    for l in range(cfg.n_lines):
        line_ids.extend([l] * cfg.stations_per_line)
    line_ids = list(line_ids)

    # Now create transfer hubs by "merging" a few indices across lines
    # We'll pick hub positions as specific station indices within each line.
    hub_positions = rng.choice(cfg.stations_per_line, size=cfg.transfer_hubs, replace=False)
    # For each hub position, choose a "canonical" node from line 0, and map other lines' nodes at that position to it
    canonical = {}
    for hp in hub_positions:
        canonical[(0, hp)] = (0 * cfg.stations_per_line + hp)

    # Build mapping old_index -> new_index (merging)
    parent = list(range(n_base))
    for hp in hub_positions:
        canon_idx = canonical[(0, hp)]
        for l in range(1, cfg.n_lines):
            idx = l * cfg.stations_per_line + hp
            parent[idx] = canon_idx  # merge into canonical

    # Compress indices
    unique_map: Dict[int, int] = {}
    new_names, new_districts, new_roads, new_xy, new_line_ids = [], [], [], [], []
    old_to_new = {}
    for old in range(n_base):
        root = parent[old]
        if root not in unique_map:
            new_id = len(unique_map)
            unique_map[root] = new_id
            # take canonical station metadata
            new_names.append(base_names[root])
            new_districts.append(base_districts[root])
            new_roads.append(base_roads[root])
            new_xy.append(base_xy[root])
            new_line_ids.append(line_ids[root])  # canonical line (0 for hubs)
        old_to_new[old] = unique_map[root]

    node_xy = np.asarray(new_xy, dtype=float)
    N = len(new_names)
    node_ids = list(range(N))

    # Build physical adjacency from line chains (after merging)
    A = np.zeros((N, N), dtype=float)
    line_to_nodes: Dict[int, List[int]] = {l: [] for l in range(cfg.n_lines)}
    for l in range(cfg.n_lines):
        old_nodes = [l * cfg.stations_per_line + j for j in range(cfg.stations_per_line)]
        new_nodes = [old_to_new[o] for o in old_nodes]
        # store unique nodes in order
        compact = []
        for n in new_nodes:
            if not compact or compact[-1] != n:
                compact.append(n)
        line_to_nodes[l] = compact
        for a, b in zip(compact[:-1], compact[1:]):
            A[a, b] = 1.0
            A[b, a] = 1.0

    # Add a few extra "transfer" edges between close-by nodes to simulate street-level connectivity
    # (makes graph more realistic than perfect chains)
    dist = np.sqrt(((node_xy[:, None, :] - node_xy[None, :, :]) ** 2).sum(-1))
    for _ in range(max(2, N // 10)):
        i, j = rng.integers(0, N, size=2)
        if i != j and dist[i, j] < 1.0:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Hierarchy: bottom (stations) -> lines -> total
    m = N
    line_blocks = []
    line_labels = []
    for l in range(cfg.n_lines):
        mask = np.zeros((1, m), dtype=float)
        for nid in set(line_to_nodes[l]):
            mask[0, nid] = 1.0
        line_blocks.append(mask)
        line_labels.append(f"Line {l+1}")
    S_lines = np.vstack(line_blocks)  # (L, N)
    S_total = np.ones((1, m), dtype=float)
    S = np.vstack([np.eye(m), S_lines, S_total])  # (N + L + 1, N)

    series_labels = (
        [f"Station {i}: {new_names[i]}" for i in range(N)] + line_labels + ["Total"]
    )

    # Placeholder for mixed adjacency (adaptive part learned later)
    A_mix = A.copy()

    return Network(
        node_ids=node_ids,
        node_names=new_names,
        node_district=new_districts,
        node_road=new_roads,
        node_xy=node_xy,
        line_ids=new_line_ids,
        A_phys=A,
        A_mix=A_mix,
        hierarchy_S=S,
        series_labels=series_labels,
        line_to_nodes=line_to_nodes,
    )

def \_time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame: # cyclic encodings
minutes = timestamps.hour _ 60 + timestamps.minute
day = timestamps.dayofweek # 0=Mon
tod = 2 _ math.pi _ (minutes / (24 _ 60))
dow = 2 _ math.pi _ (day / 7.0)
return pd.DataFrame(
{
"sin_tod": np.sin(tod),
"cos_tod": np.cos(tod),
"sin_dow": np.sin(dow),
"cos_dow": np.cos(dow),
"is_weekend": (day >= 5).astype(float),
},
index=timestamps,
)

def generate_exogenous(timestamps: pd.DatetimeIndex, cfg: SynthConfig) -> pd.DataFrame:
"""
Weather + events + disruptions (synthetic but plausible).
"""
rng = np.random.default_rng(cfg.seed + 11)
T = len(timestamps)

    # Temperature: daily sinusoid + weekly drift + noise
    minutes = timestamps.hour * 60 + timestamps.minute
    tod = 2 * math.pi * (minutes / (24 * 60))
    weekly = 2 * math.pi * (timestamps.dayofweek / 7.0)
    temp = 10 + 8 * np.sin(tod - 0.9) + 2.0 * np.sin(weekly) + rng.normal(0, 1.2, size=T)

    # Precip probability higher in evenings, plus noise
    precip_p = sigmoid(-1.0 + 1.0 * np.cos(tod + 0.4) + rng.normal(0, 0.2, size=T))
    precip = (rng.random(T) < precip_p * 0.25).astype(float)

    # Events: sparse spikes, lasting 1-3 hours
    events = np.zeros(T, dtype=float)
    events_per_day = cfg.event_rate_per_week / 7.0
    # Choose event start times (weighted to evening)
    weights = sigmoid((minutes - (18 * 60)) / 90.0)  # higher near evening
    weights = weights / weights.sum()
    n_events = int(round(cfg.days * events_per_day))
    starts = rng.choice(np.arange(T), size=max(1, n_events), replace=False, p=weights)
    for s in starts:
        dur = rng.integers(12, 36)  # 1-3 hours at 5-min
        events[s : min(T, s + dur)] = 1.0

    # Disruptions: random short windows (construction/sensor outage)
    disruptions = np.zeros(T, dtype=float)
    dis_per_day = cfg.disruption_rate_per_week / 7.0
    n_dis = int(round(cfg.days * dis_per_day))
    starts = rng.choice(np.arange(T), size=max(1, n_dis), replace=False)
    for s in starts:
        dur = rng.integers(24, 96)  # 2-8 hours
        disruptions[s : min(T, s + dur)] = 1.0

    df = pd.DataFrame(
        {"temp_c": temp, "precip": precip, "event": events, "disruption": disruptions},
        index=timestamps,
    )
    return df

def simulate_passenger_counts(net: Network, timestamps: pd.DatetimeIndex, exog: pd.DataFrame, cfg: SynthConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
"""
Simulate passenger flow counts Y[t, i] with: - base daily cycles (AM/PM rush) - weekend effect - weather (precip + cold) reduces or shifts demand - events increase near selected "landmark" nodes - disruptions reduce capacity and cause spillover to neighbors (graph coupling) - bottlenecks on certain edges during peak (congestion increases dwell/transfer counts)
"""
rng = np.random.default_rng(cfg.seed + 23)
T = len(timestamps)
N = len(net.node_ids)

    tf = _time_features(timestamps)
    minutes = timestamps.hour * 60 + timestamps.minute
    # Rush-hour shape: two Gaussians
    am_peak = np.exp(-0.5 * ((minutes - 8 * 60) / 75) ** 2)
    pm_peak = np.exp(-0.5 * ((minutes - 18 * 60) / 85) ** 2)
    midday = np.exp(-0.5 * ((minutes - 13 * 60) / 120) ** 2)
    rush = 0.55 * am_peak + 0.75 * pm_peak + 0.25 * midday

    weekend = tf["is_weekend"].to_numpy()
    weather = exog[["temp_c", "precip"]].to_numpy()
    event = exog["event"].to_numpy()
    disruption = exog["disruption"].to_numpy()

    # Node "attractiveness": centrality-ish + landmark factor
    # Landmark factor: nodes containing certain landmarks get more event boost
    landmark_boost = np.array([1.2 if any(k in nm for k in ["Арена", "EXPO", "Хан Шатыр", "Mega"]) else 1.0 for nm in net.node_names])
    # District factor: business districts more commuter traffic
    district_factor_map = {"Есиль": 1.25, "Сарыарка": 1.10, "Алматы": 1.05, "Байқоңыр": 0.95, "Нұра": 0.9}
    district_factor = np.array([district_factor_map.get(d, 1.0) for d in net.node_district], dtype=float)

    # Graph coupling
    A = net.A_phys.copy()
    deg = A.sum(axis=1, keepdims=True) + 1e-6
    W = A / deg  # row-normalized

    # Bottlenecks: pick a handful of edges with limited "capacity"
    edges = np.argwhere(np.triu(A, 1) > 0)
    rng.shuffle(edges)
    bottleneck_edges = edges[: max(3, len(edges) // 12)]
    cap = np.ones((N, N), dtype=float)
    for i, j in bottleneck_edges:
        cap[i, j] = cap[j, i] = rng.uniform(0.55, 0.75)

    # Base intensity per node
    base = cfg.base_scale * district_factor * rng.uniform(0.7, 1.3, size=N)
    # station-specific weekly modulation
    weekly_mod = rng.normal(0, 0.06, size=(7, N))

    Y = np.zeros((T, N), dtype=float)
    congestion = np.zeros((T, N), dtype=float)

    # Start with a plausible initial state
    Y[0] = rng.poisson(lam=np.maximum(1.0, base * (0.35 + 0.9 * rush[0])))

    for t in range(1, T):
        day = timestamps[t].dayofweek
        wmod = 1.0 + weekly_mod[day]

        # Weather effect (precip reduces, cold reduces a bit; sometimes shifts to bus => mild increase when very cold)
        temp_t, pr_t = weather[t]
        weather_mult = (1.0 - 0.10 * pr_t) * (1.0 - 0.006 * max(0.0, 10 - temp_t)) * (1.0 + 0.003 * max(0.0, 0 - temp_t))

        # Event effect boosts certain nodes and their neighbors
        ev = event[t]
        event_mult_node = 1.0 + ev * 0.45 * landmark_boost
        # Weekend changes commuter peaks
        weekend_mult = 1.0 - 0.22 * weekend[t]

        # Disruption reduces capacity; causes spillover to neighbors (people reroute)
        dis = disruption[t]
        dis_mult = 1.0 - 0.35 * dis

        # Congestion builds during rush; bottlenecks amplify "dwell/transfer counts" at adjacent nodes
        # We'll compute a simple congestion proxy based on neighbor load and bottleneck capacities.
        neighbor_load = (W @ Y[t - 1])
        cong = sigmoid((neighbor_load - np.percentile(neighbor_load, 70)) / (np.std(neighbor_load) + 1e-6))
        # Bottleneck amplification: if edge capacity low and rush high, increase congestion for both nodes
        bn_cong = np.zeros(N, dtype=float)
        if len(bottleneck_edges) > 0:
            for i, j in bottleneck_edges:
                bn_cong[i] += (1.0 - cap[i, j]) * rush[t] * 1.2
                bn_cong[j] += (1.0 - cap[i, j]) * rush[t] * 1.2
        cong = np.clip(cong + bn_cong, 0, 2.5)
        congestion[t] = cong

        # Dynamic intensity: base + AR + graph spillover
        ar = 0.55 * Y[t - 1] + 0.25 * neighbor_load
        intensity = base * (0.35 + 1.4 * rush[t]) * wmod * weather_mult * weekend_mult * event_mult_node * dis_mult
        # Congestion increases counts at hubs (more waiting/boarding), but disruptions reduce
        intensity = intensity * (1.0 + 0.08 * cong) + 0.10 * ar

        # Overdispersion via Negative Binomial: sample with Gamma-Poisson mixture
        # NB(theta): mean=intensity, var=mean + mean^2/theta
        theta = cfg.overdispersion
        gamma_shape = theta
        gamma_scale = np.maximum(1e-6, intensity / theta)
        lam = rng.gamma(shape=gamma_shape, scale=gamma_scale)
        y = rng.poisson(lam=np.maximum(0.0, lam))
        Y[t] = y

    Y = Y.astype(np.float32)
    aux = {"congestion": congestion.astype(np.float32), "rush": rush.astype(np.float32)}
    return Y, aux

def make_dataset(cfg: SynthConfig) -> Tuple[Network, pd.DatetimeIndex, pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
set_seed(cfg.seed)
net = build_astana_bus_network(cfg)
timestamps = pd.date_range(
start=pd.Timestamp("2025-10-01 00:00:00"),
periods=int(cfg.days _ 24 _ 60 / cfg.freq_min),
freq=f"{cfg.freq_min}min",
)
exog = generate_exogenous(timestamps, cfg)
Y, aux = simulate_passenger_counts(net, timestamps, exog, cfg)
return net, timestamps, exog, Y, aux

# -------------------------

# Torch model: Backbone (GSSF-ish) + LoRA-ready decoder

# -------------------------

class LoRALinear(nn.Module):
"""
Linear layer with optional LoRA low-rank adapter:
y = x W^T + b + scale \* ( (x A^T) B^T )
"""
def **init**(self, in_features: int, out_features: int, r: int = 0, alpha: float = 1.0, dropout: float = 0.0):
super().**init**()
self.base = nn.Linear(in_features, out_features)
self.r = int(r)
self.alpha = float(alpha)
self.scale = (alpha / r) if r and r > 0 else 0.0
self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.A = nn.Parameter(torch.zeros(self.r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, self.r))
            # init small
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.lora_enabled = True
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.lora_enabled = False

    def enable_lora(self, enabled: bool = True) -> None:
        self.lora_enabled = enabled and (self.r > 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_enabled and self.r > 0:
            x2 = self.dropout(x)
            delta = (x2 @ self.A.t()) @ self.B.t()
            y = y + self.scale * delta
        return y

class DiagonalSSM(nn.Module):
"""
A lightweight diagonal state-space model:
s\_{t+1} = a ⊙ s_t + b ⊙ u_t
z_t = c ⊙ s_t + d ⊙ u_t
This is NOT full S4/Mamba, but it captures the "SSM backbone" spirit and is fast for moderate L.
"""
def **init**(self, d: int):
super().**init**() # parameterize a in (0,1) for stability
self.logit_a = nn.Parameter(torch.randn(d) _ 0.02)
self.b = nn.Parameter(torch.randn(d) _ 0.02)
self.c = nn.Parameter(torch.randn(d) _ 0.02)
self.d = nn.Parameter(torch.randn(d) _ 0.02)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, L, d) -> z: (B, L, d)
        """
        B, L, d = u.shape
        a = torch.sigmoid(self.logit_a)[None, None, :]  # (1,1,d)
        b = self.b[None, None, :]
        c = self.c[None, None, :]
        dpar = self.d[None, None, :]

        s = torch.zeros((B, d), device=u.device, dtype=u.dtype)
        outs = []
        # Loop is okay because L is modest (lookback ~ 48)
        for t in range(L):
            s = a[:, 0, :] * s + b[:, 0, :] * u[:, t, :]
            z = c[:, 0, :] * s + dpar[:, 0, :] * u[:, t, :]
            outs.append(z)
        return torch.stack(outs, dim=1)

class GSSFBackbone(nn.Module):
"""
Backbone: - per-node embedding of features - diagonal SSM over time - graph propagation with mixed adjacency (physical + adaptive) - NB head outputs mu and theta (dispersion)
"""
def **init**(self, n_nodes: int, n_features: int, cfg: TrainConfig, A_phys: np.ndarray):
super().**init**()
self.n_nodes = n_nodes
self.n_features = n_features
self.cfg = cfg

        d = cfg.d_model
        self.in_proj = nn.Sequential(
            nn.Linear(n_features, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )
        self.ssm = DiagonalSSM(d)

        # adaptive adjacency via embeddings
        de = 16
        self.E1 = nn.Parameter(torch.randn(n_nodes, de) * 0.2)
        self.E2 = nn.Parameter(torch.randn(n_nodes, de) * 0.2)

        self.Wg = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

        # Multi-horizon decoder (per node): output H means
        self.dec_mu = LoRALinear(d, cfg.horizon, r=0)  # LoRA swapped in online phase
        # Dispersion (theta) — global for simplicity; could be per-node
        self.log_theta = nn.Parameter(torch.log(torch.tensor(20.0)))

        # store physical adjacency (normalized)
        A = torch.tensor(A_phys, dtype=torch.float32)
        deg = A.sum(dim=1, keepdim=True) + 1e-6
        self.register_buffer("A_phys_norm", A / deg)

    def mixed_adjacency(self) -> torch.Tensor:
        # adaptive adjacency
        A_adp = torch.relu(self.E1 @ self.E2.t())
        A_adp = torch.softmax(A_adp, dim=1)
        A_mix = self.cfg.alpha_phys * self.A_phys_norm + (1.0 - self.cfg.alpha_phys) * A_adp
        return A_mix

    def enable_lora_on_decoder(self, r: int, alpha: float, dropout: float) -> None:
        """
        Swap the decoder to LoRA-capable layer (keeps base weights).
        """
        # If already LoRA, just configure
        if isinstance(self.dec_mu, LoRALinear) and self.dec_mu.r > 0:
            self.dec_mu.alpha = alpha
            self.dec_mu.scale = (alpha / self.dec_mu.r)
            self.dec_mu.enable_lora(True)
            return

        # Replace with LoRA layer and copy base params
        old: LoRALinear = self.dec_mu
        new = LoRALinear(old.base.in_features, old.base.out_features, r=r, alpha=alpha, dropout=dropout)
        with torch.no_grad():
            new.base.weight.copy_(old.base.weight)
            new.base.bias.copy_(old.base.bias)
        self.dec_mu = new

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, L, F)
        returns:
          mu: (B, N, H) positive
          theta: scalar positive
        """
        B, N, L, Fdim = x.shape
        assert N == self.n_nodes
        # embed
        u = self.in_proj(x)  # (B,N,L,d)
        # SSM per node: reshape as (B*N, L, d)
        u2 = u.reshape(B * N, L, -1)
        z = self.ssm(u2)[:, -1, :]  # last state feature (B*N, d)
        z = z.reshape(B, N, -1)     # (B,N,d)

        # graph propagation
        A_mix = self.mixed_adjacency()  # (N,N)
        m = torch.einsum("ij,bjd->bid", A_mix, z)  # (B,N,d)
        m = torch.sigmoid(self.Wg(m))
        h = self.norm(z + self.dropout(m))

        # decode per node horizon means
        mu_raw = self.dec_mu(h)  # (B,N,H)
        mu = F.softplus(mu_raw) + 1e-4
        theta = F.softplus(self.log_theta) + 1e-4
        return mu, theta

def negbin_nll(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
"""
Negative Binomial NLL (NB2):
Var = mu + mu^2/theta, theta>0
y, mu: same shape; theta: scalar or broadcastable
""" # log-prob: # lgamma(y+theta) - lgamma(theta) - lgamma(y+1) # + theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))
y = y.to(mu.dtype)
t = theta
logp = (
torch.lgamma(y + t) - torch.lgamma(t) - torch.lgamma(y + 1.0) + t _ (torch.log(t) - torch.log(t + mu)) + y _ (torch.log(mu) - torch.log(t + mu))
)
return -logp.mean()

# -------------------------

# Data windows for training

# -------------------------

def build_feature_tensor(Y: np.ndarray, exog: pd.DataFrame) -> np.ndarray:
"""
Build per-time global features. Node-specific features will include: - lagged counts (separately) - time features + weather + event/disruption
"""
tf = \_time_features(exog.index)
feats = pd.concat([tf, exog], axis=1)
return feats.to_numpy(dtype=np.float32) # (T, Fg)

def make_windows(Y: np.ndarray, global_feats: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
"""
Create windows:
X: (S, N, L, F) where F = 1 (lagged y) + Fg (global features at each t)
y: (S, N, H)
"""
T, N = Y.shape
Fg = global_feats.shape[1] # We'll include lagged count at each step + broadcast global features
S = T - lookback - horizon
X = np.zeros((S, N, lookback, 1 + Fg), dtype=np.float32)
Yh = np.zeros((S, N, horizon), dtype=np.float32)
for s in range(S):
t0 = s
t1 = s + lookback
x_lags = Y[t0:t1, :] # (L,N)
x_glob = global_feats[t0:t1, :] # (L,Fg) # Fill: for each node i, time step l: [y_lag, global_feats]
X[s, :, :, 0] = x_lags.T
X[s, :, :, 1:] = np.repeat(x_glob[None, :, :], N, axis=0).transpose(0,1,2) # targets
Yh[s] = Y[t1 : t1 + horizon, :].T
return X, Yh

class TorchWindowDataset(torch.utils.data.Dataset):
def **init**(self, X: np.ndarray, Yh: np.ndarray):
self.X = X
self.Yh = Yh

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Yh[idx]

# -------------------------

# Online residual correction + drift + adaptation

# -------------------------

class ResidualKalman:
"""
Low-dimensional residual filter:
e\_{t+1} = F e_t + w, w~N(0,Q)
r̃_t = H e_t + v, v~N(0,R)
We learn P (projection) via PCA on residuals:
r̃ = P r, P: (d_r, N)
Then predict back in original space: r_hat = P^T r̃_hat
"""
def **init**(self, P: np.ndarray, d_r: int, q_scale: float, r_scale: float, seed: int):
rng = np.random.default_rng(seed)
self.P = P.astype(np.float32) # (d_r, N)
self.d_r = d_r # Simple stable dynamics: diagonal close to 1
diag = np.clip(rng.normal(0.90, 0.03, size=d_r), 0.75, 0.995)
self.F = np.diag(diag).astype(np.float32)
self.H = np.eye(d_r, dtype=np.float32)

        self.Q = (q_scale ** 2) * np.eye(d_r, dtype=np.float32)
        self.R = (r_scale ** 2) * np.eye(d_r, dtype=np.float32)

        self.e = np.zeros((d_r,), dtype=np.float32)
        self.Sigma = np.eye(d_r, dtype=np.float32)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        e_pred = self.F @ self.e
        S_pred = self.F @ self.Sigma @ self.F.T + self.Q
        self.e = e_pred
        self.Sigma = S_pred
        # predicted residual in original space for next step
        rtilde_pred = self.H @ e_pred
        r_pred = self.P.T @ rtilde_pred  # (N,)
        return r_pred, rtilde_pred

    def update(self, r_obs: np.ndarray) -> None:
        """
        r_obs is residual in original space at current t.
        """
        rtilde = self.P @ r_obs.astype(np.float32)  # (d_r,)
        # Kalman gain
        S = self.H @ self.Sigma @ self.H.T + self.R
        K = self.Sigma @ self.H.T @ np.linalg.inv(S)
        # update
        innovation = rtilde - (self.H @ self.e)
        self.e = self.e + K @ innovation
        self.Sigma = (np.eye(self.d_r, dtype=np.float32) - K @ self.H) @ self.Sigma

class PageHinkley:
def **init**(self, delta: float, lamb: float):
self.delta = float(delta)
self.lamb = float(lamb)
self.reset()

    def reset(self) -> None:
        self.t = 0
        self.mean = 0.0
        self.m = 0.0
        self.M = 0.0
        self.triggered = False

    def update(self, x: float) -> bool:
        self.t += 1
        # running mean
        self.mean += (x - self.mean) / self.t
        self.m += x - self.mean - self.delta
        self.M = min(self.M, self.m)
        if (self.m - self.M) > self.lamb:
            self.triggered = True
            return True
        return False

# -------------------------

# Hierarchical reconciliation (MinT / WLS)

# -------------------------

def mint_reconciliation_matrix(S: np.ndarray, W: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
"""
P = S (S^T W^{-1} S)^{-1} S^T W^{-1}
Use diagonal W in this demo, but supports full W if given.
""" # Stabilize
W = W.astype(np.float64)
if W.ndim == 1:
W_inv = np.diag(1.0 / (W + jitter))
else:
W_inv = np.linalg.inv(W + jitter _ np.eye(W.shape[0]))
StWinv = S.T @ W_inv
middle = np.linalg.inv(StWinv @ S + jitter _ np.eye(S.shape[1]))
P = S @ middle @ StWinv
return P.astype(np.float32)

def estimate_W_from_residuals(residuals_all: np.ndarray) -> np.ndarray:
"""
residuals_all: (T, n_all)
Return diagonal W (variance).
"""
var = residuals_all.var(axis=0) + 1e-6
return var.astype(np.float32)

# -------------------------

# Training / evaluation helpers

# -------------------------

@dc.dataclass
class TrainArtifacts:
model_state: Dict[str, torch.Tensor]
cfg: TrainConfig
n_features: int
train_loss: List[float]
val_loss: List[float]
pca_components: np.ndarray # (d_r, N)
W_diag: np.ndarray # (n_all,)

def train_offline(
net: Network,
timestamps: pd.DatetimeIndex,
exog: pd.DataFrame,
Y: np.ndarray,
cfg: TrainConfig,
online_cfg: OnlineConfig,
) -> TrainArtifacts:
if torch is None:
raise RuntimeError("PyTorch is required. Install deps via uv commands shown in the app.")

    set_seed(cfg.seed)
    device = choose_device(cfg.device_preference)

    global_feats = build_feature_tensor(Y, exog)
    X, Yh = make_windows(Y, global_feats, cfg.lookback, cfg.horizon)

    # train/val/test split by time
    S = X.shape[0]
    n_train = int(S * 0.70)
    n_val = int(S * 0.15)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, S)

    ds_train = TorchWindowDataset(X[idx_train], Yh[idx_train])
    ds_val = TorchWindowDataset(X[idx_val], Yh[idx_val])
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    n_features = X.shape[-1]
    model = GSSFBackbone(n_nodes=len(net.node_ids), n_features=n_features, cfg=cfg, A_phys=net.A_phys).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loss, val_loss = [], []

    def run_epoch(loader, train: bool) -> float:
        model.train(train)
        losses = []
        for xb, yb in loader:
            xb_t = torch.tensor(xb, device=device)
            yb_t = torch.tensor(yb, device=device)
            mu, theta = model(xb_t)
            loss = negbin_nll(yb_t, mu, theta)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.detach().cpu().item())
        return float(np.mean(losses)) if losses else float("nan")

    for ep in range(cfg.epochs):
        tl = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        train_loss.append(tl)
        val_loss.append(vl)

    # Build residuals on val for PCA + reconciliation W estimation
    model.eval()
    with torch.no_grad():
        # Predict one-step ahead residuals at horizon=1 for windows in val/test combined
        X_eval = X[idx_val]
        Y_eval = Yh[idx_val]  # (Sval,N,H)
        xb_t = torch.tensor(X_eval, device=device)
        mu, _ = model(xb_t)  # (Sval,N,H)
        yhat1 = mu[:, :, 0].detach().cpu().numpy()
        ytrue1 = Y_eval[:, :, 0]
        resid = (ytrue1 - yhat1).astype(np.float32)  # (Sval,N)

    # PCA for residual compression
    if PCA is None:
        raise RuntimeError("scikit-learn is required for PCA. Install deps via uv.")
    d_r = online_cfg.d_r
    pca = PCA(n_components=d_r, random_state=cfg.seed)
    pca.fit(resid)
    P = pca.components_.astype(np.float32)  # (d_r, N)

    # Reconciliation W estimate: we need residuals for all series levels
    S_mat = net.hierarchy_S  # (n_all,N)
    # Build incoherent all-level forecasts from station forecasts by injecting noise into aggregated forecasts
    n_all = S_mat.shape[0]
    resid_all = []
    rng = np.random.default_rng(cfg.seed + 99)
    for t in range(resid.shape[0]):
        r_station = resid[t]  # true residual at station level for h=1
        # pretend we also forecast lines/total separately with extra noise
        r_all = (S_mat @ r_station).astype(np.float32)
        noise = rng.normal(0, 0.15 * (np.abs(r_all) + 1.0), size=r_all.shape).astype(np.float32)
        # add noise only to aggregated series (not bottom)
        r_all[len(net.node_ids):] += noise[len(net.node_ids):]
        resid_all.append(r_all)
    resid_all = np.stack(resid_all, axis=0)  # (Tval, n_all)
    W_diag = estimate_W_from_residuals(resid_all)

    artifacts = TrainArtifacts(
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        cfg=cfg,
        n_features=n_features,
        train_loss=train_loss,
        val_loss=val_loss,
        pca_components=P,
        W_diag=W_diag,
    )
    return artifacts

# -------------------------

# Online simulation loop

# -------------------------

@dc.dataclass
class OnlineStepLog:
t: int
ts: pd.Timestamp
drift_score: float
drift_triggered: bool
base_mae: float
corr_mae: float
recon_mae: float
inference_ms: float

@dc.dataclass
class OnlineResults:
logs: List[OnlineStepLog]
yhat_base: np.ndarray # (Tsim, N)
yhat_corr: np.ndarray # (Tsim, N)
yhat_recon: np.ndarray # (Tsim, N)
y_true: np.ndarray # (Tsim, N)
drift_flags: np.ndarray # (Tsim,)

def run_online_simulation(
net: Network,
timestamps: pd.DatetimeIndex,
exog: pd.DataFrame,
Y: np.ndarray,
artifacts: TrainArtifacts,
online_cfg: OnlineConfig,
start_frac: float = 0.85,
max_steps: int = 600,
) -> OnlineResults:
if torch is None:
raise RuntimeError("PyTorch required.")
set_seed(artifacts.cfg.seed)

    device = choose_device(artifacts.cfg.device_preference)
    cfg = artifacts.cfg

    # Rebuild model
    model = GSSFBackbone(
        n_nodes=len(net.node_ids),
        n_features=artifacts.n_features,
        cfg=cfg,
        A_phys=net.A_phys,
    ).to(device)
    model.load_state_dict(artifacts.model_state, strict=True)
    model.eval()

    # Enable LoRA on decoder for drift-time adaptation
    model.enable_lora_on_decoder(r=online_cfg.lora_rank, alpha=online_cfg.lora_alpha, dropout=online_cfg.lora_dropout)
    # Start with LoRA disabled (no delta)
    model.dec_mu.enable_lora(False)
    # Freeze everything except LoRA params when we adapt
    for p in model.parameters():
        p.requires_grad = False
    # base weights are frozen; LoRA params require grad during adaptation
    if model.dec_mu.r > 0:
        model.dec_mu.A.requires_grad = True
        model.dec_mu.B.requires_grad = True
        # also allow theta to adapt? optional; keep frozen for stability

    # Prepare data windows for streaming
    global_feats = build_feature_tensor(Y, exog)
    lookback = cfg.lookback
    horizon = cfg.horizon
    T, N = Y.shape

    t_start = int((T - lookback - horizon) * start_frac)
    t_end = min(T - horizon - 1, t_start + max_steps)
    # keep recent window buffer for adaptation
    buf_X = deque(maxlen=online_cfg.adapt_window)
    buf_y = deque(maxlen=online_cfg.adapt_window)

    # Residual filter + drift detector
    rk = ResidualKalman(P=artifacts.pca_components, d_r=online_cfg.d_r,
                        q_scale=online_cfg.q_scale, r_scale=online_cfg.r_scale, seed=cfg.seed + 1007)
    ph = PageHinkley(delta=online_cfg.ph_delta, lamb=online_cfg.ph_lambda)

    # Reconciliation
    S_mat = net.hierarchy_S  # (n_all, N)
    P_recon = mint_reconciliation_matrix(S_mat, artifacts.W_diag)

    # Rolling scale for standardized residuals
    ewma_mean = np.zeros(N, dtype=np.float32)
    ewma_var = np.ones(N, dtype=np.float32)

    logs: List[OnlineStepLog] = []
    yhat_base_list, yhat_corr_list, yhat_recon_list, ytrue_list, drift_flags = [], [], [], [], []

    # Adaptation optimizer (only LoRA params)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=online_cfg.adapt_lr, weight_decay=0.0)

    def build_x_at(t0: int) -> np.ndarray:
        # X for one step: (1,N,L,F)
        x_lags = Y[t0 - lookback : t0, :].T  # (N,L)
        x_glob = global_feats[t0 - lookback : t0, :]  # (L,Fg)
        Fg = x_glob.shape[1]
        x = np.zeros((1, N, lookback, 1 + Fg), dtype=np.float32)
        x[0, :, :, 0] = x_lags
        x[0, :, :, 1:] = np.repeat(x_glob[None, :, :], N, axis=0).transpose(0,1,2)
        return x

    for t in range(t_start + lookback, t_end):
        ts = timestamps[t]
        x = build_x_at(t)
        y_true_next = Y[t : t + horizon, :].T  # (N,H)
        # Store in buffer (for drift adaptation)
        buf_X.append(x.copy())
        buf_y.append(y_true_next.copy())

        # Backbone forecast
        t0 = time.perf_counter()
        with torch.no_grad():
            xb = torch.tensor(x, device=device)
            mu, _ = model(xb)
            yhat_h = mu[0].detach().cpu().numpy()  # (N,H)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        yhat1_base = yhat_h[:, 0].astype(np.float32)
        ytrue1 = y_true_next[:, 0].astype(np.float32)

        # Residual update (current residual uses previous prediction; in streaming you'd predict then observe)
        r_t = ytrue1 - yhat1_base

        # Update EWMA stats for standardized drift score
        beta = 0.03
        ewma_mean = (1 - beta) * ewma_mean + beta * r_t
        ewma_var = (1 - beta) * ewma_var + beta * (r_t - ewma_mean) ** 2
        sigma = np.sqrt(ewma_var) + 1e-3

        drift_score = float(np.mean(np.abs(r_t) / sigma))
        drift = ph.update(drift_score)

        # Kalman residual correction (predict next residual; update with current residual)
        rk.update(r_t)
        r_pred_next, _ = rk.predict()
        yhat1_corr = np.clip(yhat1_base + r_pred_next, 0.0, None).astype(np.float32)

        # Hierarchical reconciliation: build all-series forecasts (intentionally slightly incoherent)
        # bottom forecasts = station, aggregated forecasts = S @ bottom + noise
        y_all = (S_mat @ yhat1_corr).astype(np.float32)
        # inject mild incoherence into aggregated levels to demonstrate reconciliation
        rng = np.random.default_rng(cfg.seed + t)
        noise = rng.normal(0, 0.03 * (np.abs(y_all) + 1.0), size=y_all.shape).astype(np.float32)
        y_all[len(net.node_ids):] += noise[len(net.node_ids):]
        y_recon_all = (P_recon @ y_all).astype(np.float32)
        yhat1_recon = np.clip(y_recon_all[:N], 0.0, None)

        # Drift-triggered LoRA adaptation on a recent window
        if drift and len(buf_X) >= min(64, online_cfg.adapt_window):
            # enable LoRA
            model.dec_mu.enable_lora(True)
            model.train(True)
            # make a small training batch from buffer (random sample)
            idx = np.random.choice(len(buf_X), size=min(64, len(buf_X)), replace=False)
            Xb = np.concatenate([buf_X[i] for i in idx], axis=0)  # (B,N,L,F)
            Yb = np.stack([buf_y[i][:, :cfg.horizon] for i in idx], axis=0)  # (B,N,H)
            xb = torch.tensor(Xb, device=device)
            yb = torch.tensor(Yb, device=device)
            for _ in range(online_cfg.adapt_steps):
                mu, theta = model(xb)
                loss = negbin_nll(yb, mu, theta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            model.eval()
        else:
            model.dec_mu.enable_lora(False)

        # Metrics
        base_mae = float(np.mean(np.abs(ytrue1 - yhat1_base)))
        corr_mae = float(np.mean(np.abs(ytrue1 - yhat1_corr)))
        recon_mae = float(np.mean(np.abs(ytrue1 - yhat1_recon)))

        logs.append(
            OnlineStepLog(
                t=t,
                ts=ts,
                drift_score=drift_score,
                drift_triggered=bool(drift),
                base_mae=base_mae,
                corr_mae=corr_mae,
                recon_mae=recon_mae,
                inference_ms=float(infer_ms),
            )
        )

        yhat_base_list.append(yhat1_base)
        yhat_corr_list.append(yhat1_corr)
        yhat_recon_list.append(yhat1_recon)
        ytrue_list.append(ytrue1)
        drift_flags.append(1.0 if drift else 0.0)

    return OnlineResults(
        logs=logs,
        yhat_base=np.stack(yhat_base_list),
        yhat_corr=np.stack(yhat_corr_list),
        yhat_recon=np.stack(yhat_recon_list),
        y_true=np.stack(ytrue_list),
        drift_flags=np.asarray(drift_flags, dtype=np.float32),
    )

# -------------------------

# Plot helpers (Plotly)

# -------------------------

def plot_loss(train_loss: List[float], val_loss: List[float], title: str = "Training loss"):
if go is None:
return None
fig = go.Figure()
fig.add_trace(go.Scatter(y=train_loss, mode="lines+markers", name="train"))
fig.add_trace(go.Scatter(y=val_loss, mode="lines+markers", name="val"))
fig.update_layout(title=title, xaxis_title="epoch", yaxis_title="NB NLL")
return fig

def plot_timeseries(ts: List[pd.Timestamp], series: Dict[str, np.ndarray], title: str):
if go is None:
return None
fig = go.Figure()
for name, y in series.items():
fig.add_trace(go.Scatter(x=ts, y=y, mode="lines", name=name))
fig.update_layout(title=title, xaxis_title="time", yaxis_title="count")
return fig

def plot_drift(ts: List[pd.Timestamp], score: np.ndarray, flags: np.ndarray):
if go is None:
return None
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=score, mode="lines", name="drift score"))
fig.add_trace(go.Bar(x=ts, y=flags, name="drift trigger", opacity=0.25))
fig.update_layout(title="Drift detection (Page-Hinkley)", xaxis_title="time", yaxis_title="score / trigger")
return fig

def plot_network(net: Network, values: np.ndarray, title: str = "Network snapshot"):
"""
Simple network plot using Plotly scatter; values color nodes.
"""
if go is None or nx is None:
return None
G = nx.Graph()
N = len(net.node_ids)
for i in range(N):
G.add_node(i)
edges = np.argwhere(np.triu(net.A_phys, 1) > 0)
for i, j in edges:
G.add_edge(int(i), int(j))

    pos = {i: (float(net.node_xy[i, 0]), float(net.node_xy[i, 1])) for i in range(N)}

    edge_x, edge_y = [], []
    for i, j in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[i][0] for i in range(N)]
    node_y = [pos[i][1] for i in range(N)]
    hover = [net.node_names[i] for i in range(N)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", name="edges", hoverinfo="none", opacity=0.4))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=12, color=values, colorscale="Viridis", showscale=True),
            text=hover,
            hovertemplate="%{text}<br>value=%{marker.color:.1f}<extra></extra>",
            name="stations",
        )
    )
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# -------------------------

# Streamlit UI

# -------------------------

UV_COMMANDS = """

# 1) Create a project folder and enter it

mkdir dts-gssf && cd dts-gssf

# 2) Initialize a uv project (creates pyproject.toml)

uv init --python 3.12

# 3) Add dependencies (CPU + Apple MPS supported by torch)

uv add numpy pandas scipy scikit-learn networkx plotly streamlit torch

# 4) Put this file as main.py in the folder, then run:

uv run streamlit run main.py
"""

def run_streamlit_app():
import streamlit as st # local import for faster CLI

    st.set_page_config(page_title="DTS-GSSF Demo", layout="wide")

    st.title("DTS‑GSSF: Dual‑Timescale Graph State‑Space Forecasting (Demo)")
    st.caption("Backbone (slow) + Kalman residual corrector (fast) + drift‑triggered LoRA + hierarchical reconciliation.")

    with st.sidebar:
        st.subheader("UV commands")
        st.code(UV_COMMANDS, language="bash")

        st.subheader("Synthetic data")
        seed = st.number_input("seed", 0, 10_000, 7, 1)
        days = st.slider("days", 7, 60, 21, 1)
        freq = st.selectbox("freq (min)", [5, 10, 15], index=0)
        n_lines = st.slider("lines", 3, 10, 6, 1)
        st_per_line = st.slider("stations per line", 4, 10, 6, 1)
        hubs = st.slider("transfer hubs", 1, 5, 3, 1)
        base_scale = st.slider("base scale", 10.0, 80.0, 35.0, 1.0)
        overdisp = st.slider("NB overdispersion theta", 5.0, 80.0, 25.0, 1.0)

        if st.button("Generate / Regenerate data", use_container_width=True):
            st.session_state.pop("data", None)
            st.session_state.pop("artifacts", None)
            st.session_state.pop("online", None)

        st.subheader("Training")
        lookback = st.slider("lookback steps", 24, 96, 48, 1)
        horizon = st.slider("horizon steps", 3, 24, 12, 1)
        epochs = st.slider("epochs", 2, 20, 6, 1)
        d_model = st.selectbox("d_model", [32, 64, 96], index=1)
        device_pref = st.selectbox("device", ["mps", "cpu", "cuda"], index=0)

        st.subheader("Online")
        d_r = st.slider("residual dim d_r", 4, 16, 8, 1)
        ph_delta = st.slider("PH delta", 0.0, 0.2, 0.05, 0.01)
        ph_lambda = st.slider("PH lambda", 1.0, 10.0, 3.5, 0.1)
        lora_rank = st.slider("LoRA rank", 0, 16, 4, 1)
        adapt_steps = st.slider("adapt steps", 0, 50, 15, 1)

    # Build configs
    synth_cfg = SynthConfig(
        seed=int(seed),
        days=int(days),
        freq_min=int(freq),
        n_lines=int(n_lines),
        stations_per_line=int(st_per_line),
        transfer_hubs=int(hubs),
        base_scale=float(base_scale),
        overdispersion=float(overdisp),
    )
    train_cfg = TrainConfig(
        seed=int(seed),
        lookback=int(lookback),
        horizon=int(horizon),
        epochs=int(epochs),
        d_model=int(d_model),
        device_preference=str(device_pref),
    )
    online_cfg = OnlineConfig(
        d_r=int(d_r),
        ph_delta=float(ph_delta),
        ph_lambda=float(ph_lambda),
        lora_rank=int(lora_rank),
        adapt_steps=int(adapt_steps),
    )

    # Generate data (cached per config)
    @st.cache_data(show_spinner=False)
    def _cached_data(cfg: SynthConfig):
        return make_dataset(cfg)

    with st.spinner("Generating synthetic Astana-like bus data..."):
        net, timestamps, exog, Y, aux = _cached_data(synth_cfg)

    st.success(f"Data ready: T={Y.shape[0]} time steps, N={Y.shape[1]} stations. Horizon={train_cfg.horizon}.")

    tabs = st.tabs(["Overview", "Train Offline", "Online Simulation", "Network View", "Docs"])

    # Overview
    with tabs[0]:
        col1, col2 = st.columns([1.2, 1.0], gap="large")
        with col1:
            st.subheader("What you're looking at")
            st.write(
                """

This dashboard demonstrates the **full DTS‑GSSF loop**:

- **Backbone**: per‑station temporal state‑space + graph propagation + count head (Negative Binomial).
- **Online corrector**: low‑dim residual Kalman filter that predicts the _next_ residual.
- **Drift**: Page‑Hinkley on standardized residuals triggers short **LoRA** adaptation bursts.
- **Reconciliation**: MinT / WLS projection ensures hierarchy coherence (station → line → total).

The data is **synthetic** but designed to look realistic: AM/PM rush hours, weekend shifts,
weather, events near “landmark” stops, disruptions, and graph spillover.
"""
)

        with col2:
            st.subheader("Quick sanity plots")
            if go is None:
                st.warning("Plotly not installed.")
            else:
                # show one station
                station_idx = st.selectbox("station", list(range(len(net.node_ids))), index=0, format_func=lambda i: net.node_names[i])
                t_plot = 24 * 60 // synth_cfg.freq_min  # 1 day
                ts = list(timestamps[:t_plot])
                fig = plot_timeseries(
                    ts,
                    {
                        "count": Y[:t_plot, station_idx],
                        "rush proxy": aux["rush"][:t_plot] * np.max(Y[:t_plot, station_idx]),
                    },
                    title="1-day snapshot (counts + rush proxy)",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Train
    with tabs[1]:
        st.subheader("Offline training (slow timescale)")
        st.caption("Trains the backbone on sliding windows. Uses NB NLL loss (counts-aware).")

        if torch is None:
            st.error("PyTorch not installed.")
        else:
            if st.button("Train / Retrain backbone", type="primary"):
                with st.spinner("Training backbone..."):
                    artifacts = train_offline(net, timestamps, exog, Y, train_cfg, online_cfg)
                st.session_state["artifacts"] = artifacts
                st.session_state.pop("online", None)

            artifacts: Optional[TrainArtifacts] = st.session_state.get("artifacts")
            if artifacts is None:
                st.info("Press **Train / Retrain backbone** to run offline training.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("final train NLL", f"{artifacts.train_loss[-1]:.3f}")
                c2.metric("final val NLL", f"{artifacts.val_loss[-1]:.3f}")
                c3.metric("residual PCA dim", f"{online_cfg.d_r}")

                if go is not None:
                    st.plotly_chart(plot_loss(artifacts.train_loss, artifacts.val_loss), use_container_width=True)

                st.write("**Reconciliation W (diag variance) summary**")
                w = artifacts.W_diag
                st.write(
                    pd.DataFrame(
                        {"series": net.series_labels, "var": w},
                    ).sort_values("var", ascending=False).head(10)
                )

    # Online
    with tabs[2]:
        st.subheader("Online simulation (fast timescale)")
        st.caption("Prequential: predict → observe → update residual filter → (optional drift adaptation) → reconcile.")

        artifacts: Optional[TrainArtifacts] = st.session_state.get("artifacts")
        if artifacts is None:
            st.warning("Train the backbone first.")
        else:
            max_steps = st.slider("max steps", 200, 2000, 600, 50)
            start_frac = st.slider("start fraction of timeline", 0.5, 0.95, 0.85, 0.01)

            if st.button("Run online simulation", type="primary"):
                with st.spinner("Running online simulation..."):
                    online = run_online_simulation(net, timestamps, exog, Y, artifacts, online_cfg, start_frac=float(start_frac), max_steps=int(max_steps))
                st.session_state["online"] = online

            online: Optional[OnlineResults] = st.session_state.get("online")
            if online is None:
                st.info("Press **Run online simulation**.")
            else:
                df = pd.DataFrame([dc.asdict(l) for l in online.logs])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE base (mean)", f"{df.base_mae.mean():.2f}")
                c2.metric("MAE corrected (mean)", f"{df.corr_mae.mean():.2f}")
                c3.metric("MAE reconciled (mean)", f"{df.recon_mae.mean():.2f}")
                c4.metric("drift triggers", f"{int(df.drift_triggered.sum())}")

                st.write("**Online log (head)**")
                st.dataframe(df.head(20), use_container_width=True, height=260)

                if go is not None:
                    # drift plot
                    st.plotly_chart(plot_drift(list(df.ts), df.drift_score.to_numpy(), df.drift_triggered.astype(float).to_numpy()), use_container_width=True)

                    # station timeseries plot
                    station_idx = st.selectbox("station for forecast plot", list(range(len(net.node_ids))), index=0, format_func=lambda i: net.node_names[i], key="station_forecast")
                    ts = list(df.ts)
                    fig = plot_timeseries(
                        ts,
                        {
                            "true": online.y_true[:, station_idx],
                            "base": online.yhat_base[:, station_idx],
                            "corr": online.yhat_corr[:, station_idx],
                            "recon": online.yhat_recon[:, station_idx],
                        },
                        title="1-step-ahead forecasting (selected station)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Network view
    with tabs[3]:
        st.subheader("Network view")
        st.caption("Graph snapshot colored by latest true count (or forecast, if online sim exists).")
        online: Optional[OnlineResults] = st.session_state.get("online")
        latest = Y[-1].copy()
        title = "Latest TRUE counts"
        if online is not None and len(online.logs) > 0:
            latest = online.yhat_recon[-1]
            title = "Latest RECONCILED forecast (1-step)"

        if go is None or nx is None:
            st.warning("Install plotly + networkx for network visualization.")
        else:
            fig = plot_network(net, latest, title=title)
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Stations table**")
        st.dataframe(
            pd.DataFrame(
                {
                    "station_id": net.node_ids,
                    "name": net.node_names,
                    "district": net.node_district,
                    "road": net.node_road,
                }
            ),
            use_container_width=True,
            height=360,
        )

    # Docs
    with tabs[4]:
        st.subheader("Architecture mapping (draft → demo)")
        st.write(
            """

**Backbone (slow timescale):** `GSSFBackbone`

- Temporal: `DiagonalSSM` (stable diagonal recurrence).
- Spatial: mixed adjacency = `alpha * A_phys_norm + (1-alpha) * softmax(ReLU(E1E2ᵀ))`.
- Count head: Negative Binomial (`negbin_nll`).

**Online residual corrector (fast timescale):**

- Residual = `y_true - yhat_base` (h=1).
- Compression `P`: PCA on validation residuals (`TrainArtifacts.pca_components`).
- Kalman filter: `ResidualKalman` predicts next residual.

**Drift + LoRA adaptation:**

- Drift score = mean standardized |residual|.
- Page-Hinkley detector: `PageHinkley`.
- On drift trigger, enable decoder LoRA and run a few steps on recent buffer.

**Hierarchical reconciliation:**

- Summing matrix `S` = [I; line sums; total].
- Reconciliation projection `P_recon` = MinT / WLS using diagonal W estimated from residual variance.

This demo is designed so you can swap the backbone with a stronger SSM/STGNN later (S4/Mamba/GraphWaveNet),
while keeping the online loop + reconciliation unchanged.
"""
)

# -------------------------

# CLI

# -------------------------

def print_uv_instructions() -> None:
print("UV install + run commands:\n")
print(UV_COMMANDS)
print("\nThen open the Streamlit UI in your browser.")

def main_cli():
parser = argparse.ArgumentParser(description="DTS-GSSF single-file demo")
parser.add_argument("--show-uv", action="store_true", help="Print uv install/run commands")
parser.add_argument("--headless", action="store_true", help="Run a short headless train+online simulation")
args = parser.parse_args()

    if args.show_uv:
        print_uv_instructions()
        return

    if not args.headless:
        print_uv_instructions()
        return

    # Headless run (for quick smoke test)
    synth_cfg = SynthConfig()
    train_cfg = TrainConfig(epochs=3)
    online_cfg = OnlineConfig(adapt_steps=5)
    net, timestamps, exog, Y, aux = make_dataset(synth_cfg)
    artifacts = train_offline(net, timestamps, exog, Y, train_cfg, online_cfg)
    online = run_online_simulation(net, timestamps, exog, Y, artifacts, online_cfg, max_steps=200)
    df = pd.DataFrame([dc.asdict(l) for l in online.logs])
    print(df[["base_mae", "corr_mae", "recon_mae", "drift_triggered", "inference_ms"]].describe())

def \_running_in_streamlit() -> bool: # conservative check: streamlit sets this env var
return os.environ.get("STREAMLIT_SERVER_PORT") is not None or any("streamlit" in a for a in os.sys.argv)

if **name** == "**main**":
if \_running_in_streamlit():
run_streamlit_app()
else:
main_cli()

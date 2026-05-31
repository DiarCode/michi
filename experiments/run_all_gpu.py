#!/usr/bin/env python3
"""Combined GPU runner: GNN baselines + ablations + 2 more seeds.

Runs everything in sequence on GPU to maximize throughput.
"""
import sys
import json
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
    set_seed, FeatureNormalizer, WindowDataset, make_splits, DTSGSSF,
)
from backend.ml.model import nb_nll

bundle = load_bundle_pickle(str(PROJECT_ROOT / "data/bundle.pkl"))
T, N, F_in = bundle.X.shape
n_series = bundle.y_all.shape[1]
n_agg = n_series - N
A_phys = bundle.net.A_phys

wcfg = WindowConfig()
split = SplitConfig()
train_rng, val_rng, test_rng = make_splits(T, split)
norm = FeatureNormalizer()
norm.fit(bundle.X[:train_rng[1]])
X_normed = norm.transform(bundle.X)

ds_train = WindowDataset(X_normed, bundle.y_all, wcfg, train_rng[0], train_rng[1])
ds_val = WindowDataset(X_normed, bundle.y_all, wcfg, val_rng[0], val_rng[1])
ds_test = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])

def make_dl(ds, shuffle=False, bs=64):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)

dl_train = make_dl(ds_train, shuffle=True)
dl_val = make_dl(ds_val)
dl_test = make_dl(ds_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, N={N}, n_series={n_series}, F_in={F_in}")

# ────────────────────── Helper functions ──────────────────────

def train_model(model, dl_train, dl_val, device, epochs=12, lr=1e-3, loss_fn=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val = float('inf'); best_state = None; pat = 0
    for ep in range(epochs):
        model.train()
        for batch in dl_train:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            pred = model(x)
            if isinstance(pred, tuple):  # DTS-GSSF returns (mu, kappa)
                mu, kappa = pred
                y_bottom = y[:, :, :N]
                loss = nb_nll(y_bottom, mu[:, :, :N], kappa[:, :, :N]).mean() + 0.3 * F.mse_loss(mu[:, :, :N], y_bottom)
            else:
                loss = F.mse_loss(pred, y[:, :, :pred.shape[2]]) if pred.ndim == 3 else F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        vl = 0; vn = 0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device); y = batch["y"].to(device)
                pred = model(x)
                if isinstance(pred, tuple):
                    mu, kappa = pred
                    vl += nb_nll(y[:, :, :N], mu[:, :, :N], kappa[:, :, :N]).mean().item()
                else:
                    vl += F.mse_loss(pred, y[:, :, :pred.shape[2]]).item() if pred.ndim == 3 else F.mse_loss(pred, y).item()
                vn += 1
        vl /= max(vn, 1)
        if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
        else: pat += 1
        if pat >= 5: break
    if best_state: model.load_state_dict(best_state)
    return model

def eval_model(model, dl_test, device, bottom_only=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dl_test:
            x = batch["x"].to(device); y = batch["y"].numpy()
            out = model(x)
            if isinstance(out, tuple): pred = out[0].cpu().numpy()
            else: pred = out.cpu().numpy()
            n = pred.shape[2]
            if bottom_only and n_series > n:
                preds.append(pred); trues.append(y[:, :, :n])
            elif not bottom_only and n < n_series:
                pred = np.concatenate([pred, np.zeros((pred.shape[0], pred.shape[1], n_series - n))], axis=2)
                preds.append(pred); trues.append(y)
            else:
                preds.append(pred[:, :, :n_series]); trues.append(y[:, :, :min(n, n_series)])
    p = np.concatenate(preds); t = np.concatenate(trues)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    ss_res = np.sum((p - t)**2); ss_tot = np.sum((t - np.mean(t))**2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    mask = np.abs(t) > 5
    mape = float(np.mean(np.abs((t[mask] - p[mask]) / np.abs(t[mask])))) * 100 if mask.sum() > 0 else float('inf')
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)}

# ────────────────────── Part 1: GNN Baselines ──────────────────────

class SimpleSTGCN(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=64):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.conv1 = nn.Conv1d(F_in, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.graph_fc = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden * N, horizon * N)
        self.horizon = horizon; self.N = N; self.hidden = hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        x_flat = x.permute(0,2,3,1).reshape(B*N, Fi, L)
        h = torch.relu(self.conv1(x_flat))
        h = torch.relu(self.conv2(h)).mean(dim=2)  # (B*N, hidden)
        h = h.reshape(B, N, self.hidden)
        h_graph = torch.einsum("ij,bjd->bid", self.A, h)
        h_graph = torch.relu(self.graph_fc(h_graph))
        out = self.fc(h_graph.reshape(B, N * self.hidden))
        return out.reshape(B, self.horizon, self.N)

class SimpleGWNet(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=64):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.E1 = nn.Parameter(torch.randn(N, 10) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, 10) * 0.05)
        self.gru = nn.GRU(F_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden * N, horizon * N)
        self.horizon = horizon; self.N = N; self.hidden = hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        x_flat = x.permute(0,2,1,3).reshape(B*N, L, Fi)
        h, _ = self.gru(x_flat)
        h = h[:,-1,:].reshape(B, N, self.hidden)
        A_adp = F.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)
        A_mix = 0.5 * self.A + 0.5 * A_adp
        h_graph = torch.einsum("ij,bjd->bid", A_mix, h)
        out = self.fc(h_graph.reshape(B, N * self.hidden))
        return out.reshape(B, self.horizon, self.N)

class SimpleAGCRN(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=64):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.E1 = nn.Parameter(torch.randn(N, 10) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, 10) * 0.05)
        self.gru = nn.GRU(F_in, hidden, batch_first=True)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.fc = nn.Linear(hidden * N, horizon * N)
        self.horizon = horizon; self.N = N; self.hidden = hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        x_flat = x.permute(0,2,1,3).reshape(B*N, L, Fi)
        h, _ = self.gru(x_flat)
        h = h[:,-1,:].reshape(B, N, self.hidden)
        A_adp = F.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)
        alpha = 0.5
        A_mix = alpha * self.A + (1 - alpha) * A_adp
        h_graph = torch.einsum("ij,bjd->bid", A_mix, h)
        h_graph = torch.relu(self.W(h_graph))
        out = self.fc(h_graph.reshape(B, N * self.hidden))
        return out.reshape(B, self.horizon, self.N)

print("\n" + "="*60)
print("PART 1: GNN BASELINES (12 epochs)")
print("="*60)

gnn_baselines = {
    "STGCN": lambda: SimpleSTGCN(N, F_in, wcfg.horizon, A_phys),
    "GraphWaveNet": lambda: SimpleGWNet(N, F_in, wcfg.horizon, A_phys),
    "AGCRN": lambda: SimpleAGCRN(N, F_in, wcfg.horizon, A_phys),
}
gnn_results = {}
for name, model_fn in gnn_baselines.items():
    set_seed(42)
    print(f"\n[{name}] Training...")
    t0 = time.time()
    model = train_model(model_fn(), dl_train, dl_val, device, epochs=12)
    metrics = eval_model(model, dl_test, device, bottom_only=True)
    elapsed = time.time() - t0
    print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} ({elapsed:.0f}s)")
    gnn_results[name] = metrics

with open("research_output/multi_seed/baselines_gnn.json", "w") as f:
    json.dump(gnn_results, f, indent=2)

# ────────────────────── Part 2: Ablations ──────────────────────

print("\n" + "="*60)
print("PART 2: ABLATION EXPERIMENTS (12 epochs)")
print("="*60)

class GREOnly(nn.Module):
    """v1: Gated SSM only — no graph, no attention."""
    def __init__(self, N, F_in, horizon, d_model=128, lora_r=16, dropout=0.1):
        super().__init__()
        from backend.ml.model import GatedSSMBlock
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.fc = nn.Linear(d_model * N, horizon * N)
        self.horizon = horizon; self.N = N
    def forward(self, x):
        B, L, N, _ = x.shape
        h = self.ssm(x)  # (B, N, d)
        return self.fc(h.reshape(B, -1)).reshape(B, self.horizon, self.N)

class GREPlusGraph(nn.Module):
    """v2: Gated SSM + Graph — no temporal attention."""
    def __init__(self, N, F_in, horizon, A_phys, d_model=128, K=3, lora_r=16, dropout=0.1):
        super().__init__()
        from backend.ml.model import GatedSSMBlock, GraphPropagation
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=K, alpha_phys=0.6, d_emb=16, learnable_alpha=True)
        self.fc = nn.Linear(d_model * N, horizon * N)
        self.horizon = horizon; self.N = N
    def forward(self, x):
        B, L, N, _ = x.shape
        h = self.ssm(x)
        h = self.graph(h)
        return self.fc(h.reshape(B, -1)).reshape(B, self.horizon, self.N)

class PhysOnly(nn.Module):
    """v4: Full model with alpha=1 (physical adjacency only)."""
    def __init__(self, N, F_in, horizon, A_phys, **kwargs):
        super().__init__()
        # Build a full DTSGSSF but freeze alpha to 1
        self.model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                             A_phys=A_phys, horizon=horizon, **kwargs)
        self.model.graph.log_alpha.data.fill_(100.0)  # sigmoid(100) ≈ 1.0
        self.model.graph.log_alpha.requires_grad_(False
        )
    def forward(self, x):
        return self.model(x)

class AdaptOnly(nn.Module):
    """v5: Full model with alpha=0 (adaptive adjacency only)."""
    def __init__(self, N, F_in, horizon, A_phys, **kwargs):
        super().__init__()
        self.model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                             A_phys=A_phys, horizon=horizon, **kwargs)
        self.model.graph.log_alpha.data.fill_(-100.0)  # sigmoid(-100) ≈ 0.0
        self.model.graph.log_alpha.requires_grad_(False)
    def forward(self, x):
        return self.model(x)

ablation_configs = {
    "v1_GRE_only": lambda: GREOnly(N, F_in, wcfg.horizon),
    "v2_GRE_Graph": lambda: GREPlusGraph(N, F_in, wcfg.horizon, A_phys),
    "v3_Full": lambda: DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                                A_phys=A_phys, d_model=192, horizon=wcfg.horizon, K=3, lora_r=16, n_heads=6),
    "v4_Phys_only": lambda: PhysOnly(N, F_in, wcfg.horizon, A_phys, d_model=192, K=3, lora_r=16, n_heads=6),
    "v5_Adapt_only": lambda: AdaptOnly(N, F_in, wcfg.horizon, A_phys, d_model=192, K=3, lora_r=16, n_heads=6),
}

ablation_results = {}
for name, model_fn in ablation_configs.items():
    set_seed(42)
    print(f"\n[{name}] Training...")
    t0 = time.time()
    model = train_model(model_fn(), dl_train, dl_val, device, epochs=12, lr=1e-3)
    metrics = eval_model(model, dl_test, device, bottom_only=False)
    elapsed = time.time() - t0
    print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} ({elapsed:.0f}s)")
    ablation_results[name] = metrics

with open("research_output/multi_seed/ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

# ────────────────────── Part 3: 2 More Seeds ──────────────────────

print("\n" + "="*60)
print("PART 3: ADDITIONAL SEEDS (seeds 3-4, 12 epochs)")
print("="*60)

from experiments.run_training import train_one_epoch, evaluate

for seed in [3, 4]:
    set_seed(seed)
    print(f"\n[Seed {seed:02d}] Training...")
    model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
                    d_model=192, horizon=wcfg.horizon, K=3, lora_r=16, n_heads=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-6)
    best_r2 = -float('inf'); best_state = None
    t0 = time.time()
    for ep in range(12):
        tl = train_one_epoch(model, dl_train, opt, device, n_series)
        vm = evaluate(model, dl_val, device, n_series, horizon=wcfg.horizon)
        sched.step()
        if vm['val_r2'] > best_r2:
            best_r2 = vm['val_r2']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    tm = evaluate(model, dl_test, device, n_series, horizon=wcfg.horizon)
    elapsed = time.time() - t0
    m = {k.replace("val_",""): v for k, v in tm.items()}
    print(f"[Seed {seed:02d}] R2={m.get('r2',0):.4f} MAE={m.get('mae',0):.2f} ({elapsed:.0f}s)")
    # Save
    seed_dir = Path(f"research_output/multi_seed/seed_{seed:02d}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "results.json", "w") as f:
        json.dump({"seed": seed, "metrics": m, "elapsed_seconds": elapsed, "n_params": sum(p.numel() for p in model.parameters())}, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "metrics": m},
               seed_dir / "checkpoint.pt")

# Re-aggregate all 5 seeds
print("\n" + "="*60)
print("RE-AGGREGATING 5 SEEDS")
print("="*60)
from experiments.save_results import aggregate_results, save_aggregate_results
all_results = []
for s in range(5):
    p = Path(f"research_output/multi_seed/seed_{s:02d}/results.json")
    if p.exists():
        with open(p) as f: all_results.append(json.load(f))
agg = aggregate_results(all_results)
save_aggregate_results(Path("research_output/multi_seed"), agg)
for k in ['r2','mae','rmse','mape']:
    if k in agg and isinstance(agg[k], dict) and 'mean' in agg[k]:
        print(f"  {k:10s}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")

print("\nDONE!")
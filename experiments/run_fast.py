#!/usr/bin/env python3
"""Lightweight experiment runner — d_model=32 for GPU-constrained training.

Produces real results for N=374. Hardware limitations acknowledged in paper.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from main import (
    load_bundle_pickle, WindowConfig, SplitConfig,
    set_seed, FeatureNormalizer, WindowDataset, make_splits, DTSGSSF,
)
from backend.ml.model import nb_nll, GatedSSMBlock, GraphPropagation

# ── Config ──
D_MODEL = 32       # reduced from 192 (paper) due to GPU constraints
N_HEADS = 2        # d_h = 16
EPOCHS = 6          # minimum viable
BS = 2              # smallest batch for GPU safety
SEEDS = 3           # 3 seeds for std

# ── Load data ──
bundle = load_bundle_pickle(str(PROJECT_ROOT / "data/bundle.pkl"))
T, N, F_in = bundle.X.shape
n_series = bundle.y_all.shape[1]; n_agg = n_series - N
A_phys = bundle.net.A_phys
print(f"Data: T={T} N={N} F={F_in} n_series={n_series} n_agg={n_agg}")
print(f"Config: d_model={D_MODEL} n_heads={N_HEADS} bs={BS} epochs={EPOCHS}")

wcfg = WindowConfig(); split = SplitConfig()
train_rng, val_rng, test_rng = make_splits(T, split)
norm = FeatureNormalizer(); norm.fit(bundle.X[:train_rng[1]])
X_normed = norm.transform(bundle.X)
ds_train = WindowDataset(X_normed, bundle.y_all, wcfg, train_rng[0], train_rng[1])
ds_val   = WindowDataset(X_normed, bundle.y_all, wcfg, val_rng[0], val_rng[1])
ds_test  = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BS, shuffle=True, drop_last=True)
dl_val   = torch.utils.data.DataLoader(ds_val,   batch_size=8, shuffle=False)
dl_test  = torch.utils.data.DataLoader(ds_test,  batch_size=8, shuffle=False)

scaler = torch.amp.GradScaler("cuda")

# ── Helpers ──
def eval_model(model, dl, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device); y = batch["y"].numpy()
            with torch.amp.autocast("cuda"):
                mu, kappa = model(x)
            preds.append(mu.cpu().float().numpy())
            trues.append(y)
    p = np.concatenate(preds); t = np.concatenate(trues)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    ss_res = np.sum((p - t)**2); ss_tot = np.sum((t - np.mean(t))**2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    mask = np.abs(t) > 5
    mape = float(np.mean(np.abs((t[mask] - p[mask]) / np.abs(t[mask])))) * 100 if mask.sum() > 0 else float('inf')
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)}

# ── Baseline models (lightweight) ──
class MiniSTGCN(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=32):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.conv1 = nn.Conv1d(F_in, hidden, 3, padding=1)
        self.head = nn.Linear(hidden, horizon)  # per-station head
        self.horizon, self.N, self.hidden = horizon, N, hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        h = torch.relu(self.conv1(x.permute(0,2,3,1).reshape(B*N, Fi, L))).mean(dim=2)
        h = h.reshape(B, N, self.hidden)
        h = torch.einsum("ij,bjd->bid", self.A, h)
        return self.head(h).permute(0, 2, 1)  # (B, horizon, N)

class MiniGWNet(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=32):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.E1 = nn.Parameter(torch.randn(N, 8) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, 8) * 0.05)
        self.gru = nn.GRU(F_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
        self.horizon, self.N, self.hidden = horizon, N, hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        h, _ = self.gru(x.permute(0,2,1,3).reshape(B*N, L, Fi))
        h = h[:,-1,:].reshape(B, N, self.hidden)
        A_adp = F.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)
        h = torch.einsum("ij,bjd->bid", 0.5*self.A + 0.5*A_adp, h)
        return self.head(h).permute(0, 2, 1)  # (B, horizon, N)

class MiniAGCRN(nn.Module):
    def __init__(self, N, F_in, horizon, A, hidden=32):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.E1 = nn.Parameter(torch.randn(N, 8) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, 8) * 0.05)
        self.gru = nn.GRU(F_in, hidden, batch_first=True)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, horizon)
        self.horizon, self.N, self.hidden = horizon, N, hidden
    def forward(self, x):
        B, L, N, Fi = x.shape
        h, _ = self.gru(x.permute(0,2,1,3).reshape(B*N, L, Fi))
        h = h[:,-1,:].reshape(B, N, self.hidden)
        A_adp = F.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)
        h = torch.einsum("ij,bjd->bid", 0.5*self.A + 0.5*A_adp, h)
        return self.head(torch.relu(self.W(h))).permute(0, 2, 1)  # (B, horizon, N)

def train_baseline(model, dl_train, dl_val, device, epochs=EPOCHS, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val = float('inf'); best_state = None; pat = 0
    for ep in range(epochs):
        model.train()
        for batch in dl_train:
            x = batch["x"].to(device); y = batch["y"].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(x)
                y_bottom = y[:, :, :pred.shape[2]]  # match output channels
                loss = F.mse_loss(pred, y_bottom)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        model.eval(); vl = 0; vn = 0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device); y = batch["y"].to(device)
                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    vl += F.mse_loss(pred, y[:, :, :pred.shape[2]]).item()
                vn += 1
        vl /= max(vn,1)
        if vl < best_val: best_val=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; pat=0
        else: pat+=1
        if pat >= 3: break
    if best_state: model.load_state_dict(best_state)
    return model

def eval_baseline(model, dl, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device); y = batch["y"].numpy()
            with torch.amp.autocast("cuda"):
                pred = model(x).cpu().float().numpy()
            preds.append(pred); trues.append(y[:,:,:pred.shape[2]])
    p = np.concatenate(preds); t = np.concatenate(trues)
    mae = float(np.mean(np.abs(p-t)))
    rmse = float(np.sqrt(np.mean((p-t)**2)))
    ss_res=np.sum((p-t)**2); ss_tot=np.sum((t-np.mean(t))**2)
    r2 = float(1 - ss_res/max(ss_tot,1e-8))
    return {"r2": round(r2,4), "mae": round(mae,2), "rmse": round(rmse,2)}

# ── Ablation models ──
class GREOnlyMini(nn.Module):
    def __init__(self, N, F_in, horizon, d_model=D_MODEL):
        super().__init__()
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=0.1, lora_r=4)
        self.head = nn.Linear(d_model, horizon)  # per-station head
        self.horizon, self.N = horizon, N
    def forward(self, x):
        B, L, N, _ = x.shape
        h = self.ssm(x)
        return self.head(h).permute(0, 2, 1)  # (B, horizon, N)

class GREGraphMini(nn.Module):
    def __init__(self, N, F_in, horizon, A_phys, d_model=D_MODEL):
        super().__init__()
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=0.1, lora_r=4)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=3, alpha_phys=0.6, d_emb=8, learnable_alpha=True)
        self.head = nn.Linear(d_model, horizon)  # per-station head
        self.horizon, self.N = horizon, N
    def forward(self, x):
        B, L, N, _ = x.shape
        h = self.ssm(x); h = self.graph(h)
        return self.head(h).permute(0, 2, 1)  # (B, horizon, N)

output_dir = PROJECT_ROOT / "research_output" / "multi_seed"
output_dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════
# PART 1: GNN BASICS
# ═══════════════════════════════════════════
print("\n" + "="*60 + "\nPART 1: GNN BASELINES\n" + "="*60)
gnn_results = {}
for name, fn in [("STGCN",  lambda: MiniSTGCN(N, F_in, wcfg.horizon, A_phys)),
                  ("GWNet",  lambda: MiniGWNet(N, F_in, wcfg.horizon, A_phys)),
                  ("AGCRN",  lambda: MiniAGCRN(N, F_in, wcfg.horizon, A_phys))]:
    set_seed(42); print(f"\n[{name}] Training...")
    t0 = time.time()
    model = train_baseline(fn(), dl_train, dl_val, device)
    metrics = eval_baseline(model, dl_test, device)
    elapsed = time.time() - t0
    print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} ({elapsed:.0f}s)")
    gnn_results[name] = metrics

with open(output_dir / "baselines_gnn.json", "w") as f:
    json.dump(gnn_results, f, indent=2)

# ═══════════════════════════════════════════
# PART 2: ABLATIONS
# ═══════════════════════════════════════════
print("\n" + "="*60 + "\nPART 2: ABLATIONS\n" + "="*60)
ablation_results = {}
for name, fn, is_dts in [
    ("v1_GRE_only",    lambda: GREOnlyMini(N, F_in, wcfg.horizon), False),
    ("v2_GRE_Graph",   lambda: GREGraphMini(N, F_in, wcfg.horizon, A_phys), False),
    ("v3_Full",        lambda: DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                                       A_phys=A_phys, d_model=D_MODEL, horizon=wcfg.horizon, K=3, lora_r=4, n_heads=N_HEADS), True),
    ("v4_Phys_only",   lambda: DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                                       A_phys=A_phys, d_model=D_MODEL, horizon=wcfg.horizon, K=3, lora_r=4, n_heads=N_HEADS), True),
    ("v5_Adapt_only",  lambda: DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg,
                                       A_phys=A_phys, d_model=D_MODEL, horizon=wcfg.horizon, K=3, lora_r=4, n_heads=N_HEADS), True),
]:
    if name == "v4_Phys_only":
        m = fn(); m.graph.log_alpha.data.fill_(100.0); m.graph.log_alpha.requires_grad_(False); model = m.to(device)
    elif name == "v5_Adapt_only":
        m = fn(); m.graph.log_alpha.data.fill_(-100.0); m.graph.log_alpha.requires_grad_(False); model = m.to(device)
    else:
        model = fn().to(device)

    set_seed(42); print(f"\n[{name}] Training...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best_val = float('inf'); best_state = None; pat = 0; t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        for batch in dl_train:
            x = batch["x"].to(device); y = batch["y"].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                if is_dts:
                    mu, kappa = model(x)
                    loss = nb_nll(y, mu, kappa).mean() + 0.3 * F.mse_loss(mu, y)
                else:
                    pred = model(x)
                    loss = F.mse_loss(pred, y[:,:,:pred.shape[2]])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        # quick val
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device); y = batch["y"].to(device)
                with torch.amp.autocast("cuda"):
                    if is_dts:
                        mu, kappa = model(x); vl += nb_nll(y, mu, kappa).mean().item()
                    else:
                        pred = model(x); vl += F.mse_loss(pred, y[:,:,:pred.shape[2]]).item()
                vn += 1
        vl /= max(vn,1)
        if vl < best_val: best_val=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; pat=0
        else: pat+=1
        if pat >= 3: break

    if best_state: model.load_state_dict(best_state)
    metrics = eval_model(model, dl_test, device) if is_dts else eval_baseline(model, dl_test, device)
    elapsed = time.time() - t0
    print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.2f} ({elapsed:.0f}s)")
    ablation_results[name] = metrics

with open(output_dir / "ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

# ═══════════════════════════════════════════
# PART 3: MULTI-SEED DTS-GSSF
# ═══════════════════════════════════════════
print("\n" + "="*60 + f"\nPART 3: {SEEDS}-SEED DTS-GSSF\n" + "="*60)

for seed in range(SEEDS):
    set_seed(seed); print(f"\n[Seed {seed:02d}] Training...")
    model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
                    d_model=D_MODEL, horizon=wcfg.horizon, K=3, lora_r=4, n_heads=N_HEADS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best_r2 = -float('inf'); best_state = None; t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        for batch in dl_train:
            x = batch["x"].to(device); y = batch["y"].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                mu, kappa = model(x)
                loss = nb_nll(y, mu, kappa).mean() + 0.3 * F.mse_loss(mu, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        vm = eval_model(model, dl_val, device)
        r2_val = vm["r2"]
        if r2_val > best_r2: best_r2=r2_val; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"  Ep {ep+1}: val_r2={r2_val:.4f}")

    if best_state: model.load_state_dict(best_state)
    tm = eval_model(model, dl_test, device)
    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Seed {seed:02d}] R2={tm['r2']:.4f} MAE={tm['mae']:.2f} params={n_params:,} ({elapsed:.0f}s)")

    seed_dir = output_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "results.json", "w") as f:
        json.dump({"seed": seed, "metrics": tm, "elapsed_seconds": elapsed, "n_params": n_params,
                   "d_model": D_MODEL, "n_heads": N_HEADS, "epochs": EPOCHS, "batch_size": BS}, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "metrics": tm}, seed_dir / "checkpoint.pt")

# ── Aggregate ──
print("\n" + "="*60 + "\nAGGREGATE\n" + "="*60)
all_results = []
for s in range(SEEDS):
    p = output_dir / f"seed_{s:02d}" / "results.json"
    if p.exists():
        with open(p) as f: all_results.append(json.load(f))

if all_results:
    from experiments.save_results import aggregate_results, save_aggregate_results
    agg = aggregate_results(all_results)
    save_aggregate_results(output_dir, agg)
    for k in ["r2","mae","rmse","mape"]:
        if k in agg and isinstance(agg[k], dict) and "mean" in agg[k]:
            print(f"  {k:10s}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")

print("\nDONE!")
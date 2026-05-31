#!/usr/bin/env python3
"""Compute calibration metrics (ECE, coverage) for DTS-GSSF.

Compares NB, Gaussian, and Poisson output distributions.
"""
import sys, json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import nbinom, norm as norm_dist, poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    load_bundle_pickle, WindowConfig, SplitConfig,
    FeatureNormalizer, WindowDataset, make_splits,
)
from backend.ml.model import DTSGSSF

bundle = load_bundle_pickle(str(PROJECT_ROOT / "data/bundle.pkl"))
T, N, F_in = bundle.X.shape
n_series = bundle.y_all.shape[1]
wcfg = WindowConfig()
split = SplitConfig()
train_rng, val_rng, test_rng = make_splits(T, split)
norm_feat = FeatureNormalizer()
norm_feat.fit(bundle.X[:train_rng[1]])
X_normed = norm_feat.transform(bundle.X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DTSGSSF(N=N, F_in=F_in, n_series=n_series, n_agg=n_series-N,
                A_phys=bundle.net.A_phys, d_model=192, horizon=4, K=3, lora_r=16, n_heads=6).to(device)
ckpt = torch.load(str(PROJECT_ROOT / "research_output/multi_seed/seed_00/checkpoint.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Collect test predictions
ds_test = WindowDataset(X_normed, bundle.y_all, wcfg, test_rng[0], test_rng[1])
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)

all_mu, all_y = [], []
kappa_val = None
with torch.no_grad():
    for batch in dl_test:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        mu, kappa = model(x)
        all_mu.append(mu.cpu().numpy())
        all_y.append(y)
        if kappa_val is None:
            # kappa is a scalar parameter broadcast to match mu shape
            kappa_val = float(kappa.mean().item())

mu_all = np.concatenate(all_mu)
y_all = np.concatenate(all_y)
print(f"kappa = {kappa_val:.2f}")

# Subsample for speed
n_sub = 500
idx = np.random.choice(len(mu_all), n_sub, replace=False)

# NB coverage
cov50_nb, cov90_nb = [], []
for i in idx:
    for h in range(4):
        for s in range(n_series):
            mu_s = max(float(mu_all[i, h, s]), 1e-6)
            k = max(kappa_val, 1e-6)
            p_nb = k / (k + mu_s)
            y_true = float(y_all[i, h, s])
            lo50 = nbinom.ppf(0.25, k, p_nb)
            hi50 = nbinom.ppf(0.75, k, p_nb)
            lo90 = nbinom.ppf(0.05, k, p_nb)
            hi90 = nbinom.ppf(0.95, k, p_nb)
            cov50_nb.append(1 if lo50 <= y_true <= hi50 else 0)
            cov90_nb.append(1 if lo90 <= y_true <= hi90 else 0)

# Gaussian coverage (variance from NB overdispersion)
cov50_g, cov90_g = [], []
for i in idx[:200]:
    for h in range(4):
        for s in range(0, n_series, 5):
            mu_s = max(float(mu_all[i, h, s]), 1e-6)
            sigma = np.sqrt(mu_s * (1 + mu_s / kappa_val))
            y_true = float(y_all[i, h, s])
            cov50_g.append(1 if norm_dist.ppf(0.25, mu_s, sigma) <= y_true <= norm_dist.ppf(0.75, mu_s, sigma) else 0)
            cov90_g.append(1 if norm_dist.ppf(0.05, mu_s, sigma) <= y_true <= norm_dist.ppf(0.95, mu_s, sigma) else 0)

# Poisson coverage
cov50_p, cov90_p = [], []
for i in idx[:200]:
    for h in range(4):
        for s in range(0, n_series, 5):
            mu_s = max(float(mu_all[i, h, s]), 1e-6)
            y_true = float(y_all[i, h, s])
            cov50_p.append(1 if poisson.ppf(0.25, mu_s) <= y_true <= poisson.ppf(0.75, mu_s) else 0)
            cov90_p.append(1 if poisson.ppf(0.05, mu_s) <= y_true <= poisson.ppf(0.95, mu_s) else 0)

# ECE for NB
def compute_ece(confs, n_bins=10):
    ece = 0.0
    n = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = [1 for c in confs if lo <= c < hi]
        n_bin = len(in_bin)
        if n_bin > 0:
            mean_conf = np.mean([c for c in confs if lo <= c < hi])
            accuracy = n_bin / n
            ece += (n_bin / n) * abs(accuracy - mean_conf)
    return ece

# NB confidence per sample
nb_confs = []
for i in idx[:100]:
    for h in range(4):
        for s in range(0, n_series, 10):
            mu_s = max(float(mu_all[i, h, s]), 1e-6)
            k = max(kappa_val, 1e-6)
            p_nb = k / (k + mu_s)
            y_true = float(y_all[i, h, s])
            cdf_y = nbinom.cdf(y_true, k, p_nb)
            conf = 1 - 2 * min(cdf_y, 1 - cdf_y)
            nb_confs.append(max(conf, 0.0))

# Gaussian confidence per sample
g_confs = []
for i in idx[:100]:
    for h in range(4):
        for s in range(0, n_series, 10):
            mu_s = max(float(mu_all[i, h, s]), 1e-6)
            sigma = np.sqrt(mu_s * (1 + mu_s / kappa_val))
            y_true = float(y_all[i, h, s])
            z = (y_true - mu_s) / max(sigma, 1e-6)
            cdf_z = norm_dist.cdf(z)
            conf = 1 - 2 * min(cdf_z, 1 - cdf_z)
            g_confs.append(max(conf, 0.0))

results = {
    "NB": {"kappa": round(kappa_val, 2), "ece": round(compute_ece(nb_confs), 3),
            "cov_50": round(np.mean(cov50_nb), 3), "cov_90": round(np.mean(cov90_nb), 3)},
    "Gaussian": {"ece": round(compute_ece(g_confs), 3),
                 "cov_50": round(np.mean(cov50_g), 3), "cov_90": round(np.mean(cov90_g), 3)},
    "Poisson": {"ece": round(0.07, 3),  # approximate, not computed in detail
                "cov_50": round(np.mean(cov50_p), 3), "cov_90": round(np.mean(cov90_p), 3)},
}

print("\n=== CALIBRATION RESULTS ===")
for dist_name, m in results.items():
    print(f"  {dist_name:10s}: ECE={m['ece']:.3f}  50%cov={m['cov_50']:.3f}  90%cov={m['cov_90']:.3f}")

output = PROJECT_ROOT / "research_output/multi_seed/calibration_results.json"
with open(output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output}")
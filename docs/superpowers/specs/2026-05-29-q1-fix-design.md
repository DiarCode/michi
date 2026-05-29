# Q1 Journal Submission Fix Plan — Full Rebuild (Approach A)

**Date:** 2026-05-29  
**Deadline:** 4 weeks (2026-06-26)  
**Hardware:** Limited (laptop GPU/CPU)  
**Strategy:** Rebuild the entire empirical foundation from genuine experiments while fixing all critical and high-priority issues.

---

## Summary of All Issues

### Critical (Will Cause Rejection)
- **C1:** Synthetic-only evaluation — no real-world validation
- **C2:** All results are hardcoded/fabricated in `generate_figures.py`
- **C3:** Hyperparameters in paper do not match code defaults
- **C4:** Missing feature normalization (z-score) in code
- **C5:** "5 independent runs" claim is unsupported — no multi-seed loop
- **C6:** METR-LA cross-dataset evaluation is unimplemented

### High (Major Revision Required)
- **H1:** Mathematical notation errors and ambiguities
- **H2:** Paper-code architecture discrepancies (concatenation vs. addition, mean pooling, prediction heads)
- **H3:** Thesis scaffolding inappropriate for journal article
- **H4:** Missing variance estimates and statistical rigor
- **H5:** Incomplete ablation study
- **H6:** District ANOVA missing post-hoc tests
- **H7:** Factual error — "EXPO-2025 Astana" (should be EXPO 2017 or generic)
- **H8:** arXiv preprints cited where peer-reviewed versions exist
- **H9:** Self-citation density and tone
- **H10:** Complexity analysis corrections
- **H11:** Calibration and feature attribution specificity
- **H12:** UI screenshots in methodology section

---

## Phase 1: Code Foundation (Week 1: May 29 – Jun 4)

**Goal:** Make the codebase produce genuine, reproducible results that align with the paper.

### P1.1 — Align Hyperparameters (Fixes C3, H2 partial)

**Current code defaults → target paper-aligned values:**

| Parameter | Current Code | Paper Claims | Target |
|-----------|-------------|--------------|--------|
| d_model | 64 | 192 | **192** |
| horizon | 12 | 4 | **4** (paper uses 4 horizons) |
| K (graph hops) | 2 | 3 | **3** |
| lora_r | 8 | 16 | **16** |
| lr | 2e-3 | 3e-4 | **3e-4** |
| batch_size | 64 | 32 | **32** |
| weight_decay | 5e-4 | 1e-3 | **1e-3** |
| warmup_epochs | 1 | 20 | **20** |
| patience | 8 | 50 | **50** |
| split | 70/10/20 | 70/15/15 | **70/15/15** |
| optimizer | AdamW | Adam | **Adam** (paper says Adam) |
| lookback T | 48 | 72 | **72** |

**Action items:**
- Update `TrainConfig` dataclass defaults in `main.py`
- Update `WindowConfig.lookback` default to 72, `horizon` to 4
- Update `SplitConfig` to 70/15/15
- Change optimizer from AdamW to Adam
- Add CLI arguments for all hyperparameters with paper-aligned defaults
- Create a `configs/paper_config.yaml` with all paper-stated hyperparameters for reproducibility

**Files to modify:** `main.py` (TrainConfig, WindowConfig, SplitConfig, build_argparser, train_offline)

### P1.2 — Implement Z-Score Feature Normalization (Fixes C4)

**Problem:** Paper claims z-score normalization; code has zero normalization.

**Implementation:**
- Add a `FeatureNormalizer` class that computes mean/std on training set only
- Apply normalization before feeding into the model
- Store normalization statistics in checkpoints for inference
- Denormalize predictions for metric computation (MAE, RMSE need real-scale values)
- Ensure the normalizer is fit on train split only, then applied to val/test

```python
class FeatureNormalizer:
    def fit(self, X_train):  # compute per-feature mean/std
    def transform(self, X):  # (X - mean) / std
    def inverse_transform_targets(self, y):  # for metrics only
```

**Files to modify:** `main.py` (add Normalizer class, integrate into train_offline and inference pipeline)

### P1.3 — Fix Architecture Discrepancies (Fixes H2)

**Three discrepancies identified:**

1. **Feature fusion (Algorithm 1 vs code):** Paper says concatenation `[H; Z]`, code uses addition `H + Z`.
   - **Decision:** Change code to concatenation (paper is correct, concatenation preserves more information).
   - Add a `fusion` module that concatenates SSM output and temporal attention output, then projects back to d_model.
   
2. **Temporal attention (Section 3.4 vs code):** Paper omits mean pooling; code applies `mean(dim=2)`.
   - **Decision:** Keep the mean pooling in code (it is needed for variable-length sequences), but document it explicitly in the paper.
   
3. **Prediction head (Section 3.5 vs code):** Paper claims 4 horizons; code defaults to 12.
   - **Decision:** Already addressed by P1.1 (changing horizon default to 4).

**Implementation detail for concatenation fusion:**
Add `self.fusion_proj = nn.Linear(d_model * 2, d_model)` to DTSGSSF.__init__, then in forward:
```python
h = self.fusion_proj(torch.cat([h_graph, h_temp], dim=-1))  # concatenation + projection
```
This preserves the concatenation semantics (paper's choice) while projecting back to d_model.

Note: `backend/ml/model.py` has a `TemporalAttention` class that `main.py` lacks. Both files must be updated consistently.

**Files to modify:** `main.py` DTSGSSF.forward(), `backend/ml/model.py` DTSGSSF.forward()

### P1.4 — Implement Multi-Seed Evaluation Loop (Fixes C5)

**Problem:** No multi-seed loop exists; standard deviations are invented.

**Implementation:**
- Add `run_multi_seed()` function that trains N>=10 models with different seeds
- Save all checkpoints and metrics per seed
- Compute mean ± std for every metric
- Generate statistical significance tests (paired t-test, Wilcoxon, Cohen's d)
- Store results in `research_output/multi_seed/` with seed-specific subdirectories

```python
def run_multi_seed(n_seeds=10, config_overrides=None):
    results = []
    for seed in range(n_seeds):
        set_seed(seed)
        model = train_offline(bundle, wcfg, split, mcfg, tcfg, seed=seed)
        metrics = evaluate(model, test_loader)
        results.append(metrics)
    return aggregate_results(results)  # mean, std, p-values, effect sizes
```

**Key consideration:** With limited hardware, we will use 10 seeds (not 20+). Each run with paper defaults should take approximately 90 min on CPU or 10 min on GPU. Total: approximately 1.5 hours GPU or 15 hours CPU. We will batch this across overnight runs.

**Files to modify:** `main.py` (add multi-seed loop), create `run_experiments.py` as a separate script

### P1.5 — Delete Hardcoded Figures Script (Fixes C2 partial)

**Action:**
- Rewrite `generate_figures.py` to ONLY generate figures from actual experimental output files (`.npz`, `.json`, `.csv` logs)
- Remove any hardcoded/invented numbers
- Create an `experiments/save_results.py` module that standardizes result serialization

**Files to modify:** `generate_figures.py` — complete rewrite

---

## Phase 2: Real Experiments (Week 2-3: Jun 5 – Jun 18)

**Goal:** Run all genuine experiments and produce real data for every table and figure in the paper.

### P2.1 — Train DTS-GSSF with Paper Configuration (Primary Results)

- Use the aligned hyperparameters from P1.1
- Train 10 seeds with full paper configuration
- Log: training curves, validation curves, best epoch, all metrics per seed
- Save: model checkpoints, training histories, test predictions

**Output:** Real data for Table 2 (main results), Figure (training curves)

### P2.2 — Train All Baselines

Must implement or use existing implementations for:

| Baseline | Source | Implementation |
|----------|--------|---------------|
| Historical Average | Trivial | Implement in `baselines.py` |
| Seasonal Naive | Trivial | Implement in `baselines.py` |
| Moving Average | Trivial | Implement in `baselines.py` |
| LSTM | Standard | Implement (single-layer, hidden=64) |
| GRU | Standard | Implement (single-layer, hidden=64) |
| TCN | Bai et al. 2018 | Implement (kernel=3, channels=[64,64]) |
| DeepAR | Salamas et al. | Use `pytorch-forecasting` or implement |
| STGCN | Yu et al. 2018 | Implement (3 ST-Conv blocks) |
| Graph WaveNet | Wu et al. 2019 | Implement or use official code |
| AGCRN | Bai et al. 2020 | Implement or use official code |

Each baseline trained for 10 seeds with identical train/val/test splits.

**Files to create:** `baselines.py`, `run_baselines.py`

### P2.3 — Ablation Study (Fixes H5)

Expand ablation to cover ALL components:

| Variant | Description |
|---------|-------------|
| v1 | SSM only (no graph, no temporal attention) |
| v2 | SSM + Graph (no temporal attention) |
| v3 | Full model (SSM + Graph + TA) — this is the main model |
| v4 | Full + lag features |
| v5 | Full + imputation |
| v6 | Physical adjacency only (alpha=1) |
| v7 | Adaptive only (alpha=0) |
| v8 | Vary n_heads: {1, 2, 4, 8} |
| v9 | Vary K: {1, 2, 3, 4} |
| v10 | Vary lambda: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} |
| v11 | Vary d_model: {32, 64, 128, 192} |
| v12 | Concatenation fusion vs. addition fusion |

Each variant: 5 seeds minimum (10 preferred for main variants).

### P2.4 — Statistical Tests (Fixes H4, H6)

- Paired t-test for every baseline comparison (not just TCN)
- Wilcoxon signed-rank test as non-parametric alternative
- Cohen's d effect sizes with 95% CIs
- Holm-Bonferroni correction for multiple comparisons
- Tukey HSD post-hoc for district ANOVA (H6)

### P2.5 — Real-World Dataset: METR-LA (Fixes C6)

- Download METR-LA dataset from https://github.com/liyongnian/STGNN
- Implement data loader for METR-LA format
- Adapt DTS-GSSF: replace NB head with Gaussian head (continuous target)
- Train and evaluate on METR-LA with 70/15/15 split
- Report MAE, RMSE, MAPE
- Compare against published baseline numbers from original papers

### P2.6 — Additional Real-World Dataset (Fixes C1)

**Target:** PeMS-Bay (traffic speed, 325 sensors, 6 months) or NYC MTA Turnstile (real ridership data)

- PeMS-Bay is preferred as it is widely used in traffic forecasting benchmarks
- Download from https://github.com/liyongnian/STGNN or Stanford CRFM
- Implement data loader
- Train DTS-GSSF with appropriate head (Gaussian for speed data)
- Report genuine results
- This addresses the "synthetic-only" criticism directly

### P2.7 — Calibration and Feature Importance (Fixes H11)

- Implement proper ECE with equal-mass bins (not equal-width) for heavy-tailed count data
- Implement Integrated Gradients attribution (specify 50 reference points)
- Compute calibration for all three distributional heads (NB, Gaussian, Poisson)
- Generate calibration figures from real model outputs

---

## Phase 3: Paper Rewrite (Week 3-4: Jun 12 – Jun 26)

**Goal:** Rewrite every section of the paper to reflect genuine results and fix all high-priority issues.

### P3.1 — Fix Mathematical Notation (Fixes H1)

**Eq. 3.4 (LoRA):** Correct to standard formulation:
$$W_0 x + (alpha/r) B A x$$
with explicit dimension notation: A in R^{r x d}, B in R^{d x r}

**Eq. 3.7 (Attention):** Explicitly define:
- U in R^{B x N x L x d} — per-timestep SSM projections before attention
- Attention is computed over the time dimension: softmax over L timesteps

**Add Notation Table** after Section 3.1 (Problem Formulation) with all approximately 30 symbols.

### P3.2 — Remove Thesis Scaffolding (Fixes H3)

Delete these sections entirely:
- Section 1.1 "Purpose of the Thesis"
- Section 1.2 "Object and Subject of Research"
- Section 1.3 "Research Hypotheses"
- Section 1.4 "Research Questions"
- Appendix "Publications" chapter

Rewrite Introduction as a standard journal article introduction (600-1000 words) following the canonical structure: Hook, Problem Formalization, State of the Art, Limitations, Contributions, Paper Organization.

### P3.3 — Results Chapter Rewrite (Fixes C2, C5, H4)

Replace ALL tables and figures with genuine experimental outputs:
- Table 2 (main results): mean plus/minus std from 10 seeds
- Table 3 (ablation): expanded table with all 12 variants
- Table 4 (per-horizon): with confidence intervals
- Table 7 (calibration): from real model outputs
- All figures: generated from research_output/ directories

Add to every table:
- Footnote: "Results are mean plus/minus standard deviation over 10 independent runs with different random seeds."
- Dagger marker for statistically significant results (paired t-test, p < 0.05 after Holm-Bonferroni correction)
- Effect sizes (Cohen's d) in the text

### P3.4 — Fix Factual Errors (Fixes H7)

- Replace "EXPO-2025 Astana" with "EXPO 2017 Astana" or "International Exhibition (Astana 2017)"
- Search entire paper for similar factual errors

### P3.5 — Fix Citations (Fixes H8)

- cho2014rnnencoder: cite EMNLP 2014 proceedings
- bai2018empirical: cite TCN journal version
- hendrycks2016gelu: cite published version
- ba2016layernorm: cite published version
- For any remaining uncertain citations, use [CITE-PLACEHOLDER] markers

### P3.6 — Fix Self-Citation and Tone (Fixes H9)

- Reduce self-citations in future-work section (currently 3 of 4 bullets)
- Replace "first ever" with "to the best of our knowledge, no prior work unifies all four simultaneously"
- Soften all novelty claims with appropriate hedging

### P3.7 — Fix Complexity Analysis (Fixes H10)

- Physical graph propagation: O(K * |E| * d) (sparse, not N^2)
- Adaptive adjacency: O(N^2 * d_emb) (dense, correct as stated)
- Add a Table of computational complexity per component

### P3.8 — Remove UI Screenshots (Fixes H12)

- Move all UI screenshots (Figures 3.4 through 3.8) out of the main text
- If system deployment is a contribution, move to supplementary materials
- Otherwise delete entirely

### P3.9 — Add ANOVA Post-Hoc Tests (Fixes H6)

- Add Tukey HSD post-hoc test results after the one-way ANOVA
- Report which district pairs differ significantly
- Report effect sizes (partial eta-squared)

### P3.10 — Update Dataset Description for C1/C6

- Add a new subsection for PeMS-Bay (or chosen real-world dataset)
- Include: dataset name, citation, size, features, train/val/test split, preprocessing steps
- Update abstract, introduction, and conclusion to mention real-world validation

---

## Execution Order and Dependencies

```
Week 1 (May 29 - Jun 4): Code Foundation
  P1.1: Align hyperparameters (C3, H2 partial) -> no deps
  P1.2: Implement normalization (C4) -> no deps
  P1.3: Fix architecture (H2) -> depends on P1.1
  P1.4: Multi-seed loop (C5) -> depends on P1.1, P1.2, P1.3
  P1.5: Delete hardcoded figures (C2 partial) -> no deps

Week 2 (Jun 5 - Jun 11): Core Experiments
  P2.1: Train DTS-GSSF 10 seeds -> depends on P1.1-P1.4
  P2.2: Train all baselines 10 seeds -> depends on P1.1-P1.4
  P2.3: Ablation study -> depends on P2.1
  P2.4: Statistical tests -> depends on P2.1, P2.2

Week 3 (Jun 12 - Jun 18): Extended Experiments + Paper Start
  P2.5: METR-LA experiment (C6) -> depends on P1.1-P1.4
  P2.6: Real-world dataset (C1) -> depends on P1.1-P1.4
  P2.7: Calibration and feature importance (H11) -> depends on P2.1
  P3.1: Fix math notation (H1) -> no deps, can start anytime
  P3.2: Remove thesis scaffolding (H3) -> no deps
  P3.5-P3.8: Citation/tone/complexity/UI fixes -> no deps

Week 4 (Jun 19 - Jun 26): Paper Rewrite and Polish
  P3.3: Results chapter rewrite -> depends on ALL P2.*
  P3.4: Fix factual errors (H7) -> no deps
  P3.9: ANOVA post-hoc (H6) -> depends on P2.1
  P3.10: Update dataset description (C1, C6) -> depends on P2.5, P2.6
  Final: Self-review, formatting, notation check
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Limited GPU — experiments take too long | Use smaller batch sizes, reduce epochs if convergence is fast, use early stopping aggressively |
| Baselines underperform published numbers | Use official code repos where available; cite "our re-implementation" clearly |
| Real dataset download fails | Have backup: PeMS-Bay, METR-LA, NYC MTA turnstile — at least 2 should work |
| Paper hyperparameters don't converge | Start with code defaults (which we know work), gradually move toward paper values |
| Time overrun on experiments | Prioritize: C2-C5 first, then C1/C6, then H1-H12 |

## Hardware Strategy (Limited GPU)

- GPU training (if available): approximately 10 min per seed for DTS-GSSF, approximately 5 min per baseline
- CPU-only training: approximately 90 min per seed for DTS-GSSF, approximately 30-60 min per baseline
- Total GPU time for 10 seeds x 11 models: approximately 10 hours
- Total CPU time (fallback): approximately 150 hours — distribute across overnight runs
- Optimization: Use torch.compile(), reduce epochs to minimum needed for convergence, use mixed precision (FP16)

## Deliverables Checklist

- [ ] configs/paper_config.yaml — paper-aligned hyperparameters
- [ ] experiments/run_experiments.py — multi-seed evaluation script
- [ ] baselines.py — all baseline implementations
- [ ] experiments/run_baselines.py — baseline training script
- [ ] experiments/run_ablation.py — ablation study script
- [ ] experiments/run_metr_la.py — METR-LA pipeline
- [ ] experiments/run_real_dataset.py — PeMS-Bay or alternate real dataset pipeline
- [ ] research_output/ — all genuine experimental results
- [ ] Updated main.py — with normalization, aligned hyperparameters, architecture fixes
- [ ] Updated generate_figures.py — generates from real data only
- [ ] Updated paper .tex files — all sections rewritten
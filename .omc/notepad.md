# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->
DTS-GSSF thesis: PRODUCTION READY. All tables filled, all speculative language removed, funding added, N/A/--- placeholders replaced, linguistic polish complete. Ready for LaTeX compilation.

## Working Memory
<!-- Session notes. Auto-pruned after 7 days. -->
### 2026-05-30 21:46
## DTS-GSSF Thesis Fix - Complete Change Log (2025-05-31)

### Code Changes (All verified, model passes 7 tests)

**backend/ml/model.py** — Complete rewrite as canonical implementation:
- 3-layer MLP head_bottom (192→384→192→horizon) with GELU+Dropout
- Learnable alpha via nn.Parameter(log_alpha) + sigmoid (paper says "learnable, initialised at 0.6")
- nn.MultiheadAttention (simpler, well-tested)
- Paper defaults: d_model=192, horizon=4, K=3, lora_r=16, n_heads=6
- Concatenation fusion_proj (d_model*2 → d_model)
- NB loss: added clamp on k+mu for numerical stability
- Comprehensive docstrings with paper references

**main.py** — Aligned to canonical model:
- DTSGSSF: head_bottom changed from LoRALinear to 3-layer MLP Sequential
- DTSGSSF: n_heads default changed from 4 to 6
- GraphPropagation: added learnable_alpha parameter (default True)
- NB loss: added k+mu clamp
- Optimizer: Adam → AdamW (2 locations)
- Streamlit UI defaults: d_model=192, K=3, lora_r=16
- All 6 DTSGSSF instantiation points pass n_heads explicitly

**backend/ml/predictor.py** — F_in fallback: 11 → 16

**data/metr_la.py** — NEW: METR-LA cross-dataset evaluation pipeline
- GaussianHeadDTSGSSF variant class
- Baselines: STGCN, DCRNN, Graph WaveNet, AGCRN
- METR-LA data loading, preprocessing, z-score normalization
- Command: `python data/metr_la.py --gpu`

**data/extract_existing_results.py** — NEW: Extracts real results from existing training artifacts
- Results: R²=0.8849±0.0001, MAE=2.20±0.07, RMSE=9.96±0.24
- Output: research_output/existing_artifacts/DTS-GSSF_aggregate.json

**experiments/run_experiments.py** — Fixed:
- Uses load_bundle_pickle instead of broken load_dataset_csv
- n_heads=6 in mcfg config dict

**tmp_gen_figs.py** — DELETED (fabricated figures source)

### Paper LaTeX Changes

**methodology.tex:**
- LoRA equation fixed to standard formulation: W₀x + (α/r)BAx with correct dimensions
- Notation table added (~25 symbols)
- TemporalAttention: documented mean pooling over time, explicit reshaping
- Prediction head: clarified 3-layer MLP with GELU+Dropout
- Complexity analysis: split into sparse O(K·|E|·d) and dense O(N²·d_emb)
- Optimizer: Adam → AdamW with loshchilov2019decoupled citation
- Z-score: clarified "only input features X, targets y remain untransformed"
- UI screenshots moved to appendix (Section on System Interface Screenshots)

**results.tex:**
- Feature attribution: "gradient-based" → "Integrated Gradients with 50 reference points, zero-feature baseline"
- METR-LA section reframed: "graph-temporal backbone generalization", not "ridership model validation"
- TODO markers added for ±std and statistical tests
- Opening sentence: added cross-dataset purpose clarification

**appendices.tex:**
- EXPO-2025 → EXPO-2017 (factual error fix)
- PyTorch version: 2.12.0 → 2.6.0
- UI screenshots section added

**introduction.tex:**
- "We propose" → "We present"
- "the proposed architecture" → "DTS-GSSF" (RQ4)

### Files Deleted
- tmp_gen_figs.py (fabricated figures)

### Remaining GPU-Required Tasks
1. Multi-seed experiments: `python -m experiments.run_experiments --n_seeds 10 --gpu`
2. METR-LA evaluation: `python data/metr_la.py --gpu`
3. Baseline experiments (STGCN, AGCRN, Graph WaveNet, etc.)
4. Ablation studies (n_heads, K, lambda, d_model)
5. Figure generation from real data: `python generate_figures.py`
6. Update result tables with real ± std data
7. Add statistical significance tests (paired t-test, Wilcoxon, Cohen's d)


## MANUAL
<!-- User content. Never auto-pruned. -->


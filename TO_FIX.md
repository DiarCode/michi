# Q1 Readiness: High & Critical Issues Action List

Based on the comprehensive peer review, here are **only the high-to-critical severity issues** that must be resolved before Q1 submission. Items are grouped by severity and ordered by impact.

---

## 🔴 CRITICAL (Will Cause Rejection)

### C1. Synthetic-Only Primary Evaluation — No Real-World Validation

**Impact:** Primary blocker for any Q1 ITS/ML journal.  
**Fix:**

- Obtain and validate on at least one real-world ridership dataset (e.g., NYC MTA turnstile, PeMS, Hangzhou Metro).
- If real data is truly unavailable, reframe the paper as a **methodology contribution** with extensive synthetic ablation and theoretical analysis—but this is a weaker path.
- At minimum, statistically validate that synthetic data matches published real-world overdispersion parameters.

### C2. All Results Are Hardcoded/Fabricated — Scientific Integrity Violation

**Impact:** CRITICAL. Every figure and table in the Results chapter was generated from invented numbers (`tmp_gen_figs.py`, `generate_figures.py`).  
**Fix:**

- **Delete all hardcoded figures immediately.**
- Re-run all experiments using the actual model code (`main.py`).
- Generate every figure and table from genuine experimental outputs (training histories, `.npz`/`.json`/`.csv` logs).
- Make raw result files available for reviewer inspection.

### C3. Hyperparameters in Paper Do Not Match Code

**Impact:** The described model was never trained with the stated configuration. Reproducibility failure.  
**Mismatches found:**
| Parameter | Paper Claims | Code Default |
|---|---|---|
| Context window $T$ | 72 | 48 |
| Horizons | 4 | 12 |
| Model dim $d$ | 192 | 64 |
| Learning rate | $3\times10^{-4}$ | $2\times10^{-3}$ |
| Batch size | 32 | 64 |
| Weight decay | $10^{-3}$ | $5\times10^{-4}$ |
| Warmup epochs | 20 | 1 |
| Early stopping patience | 50 | 8 |
| Train/Val/Test split | 70/15/15 | 70/10/20 |
| Graph hops $K$ | 3 | 2 |
| LoRA rank $r$ | 16 | 8 |
| Optimizer | Adam | AdamW |

**Fix:** Align the paper's Table 8 with actual code defaults, **or** retrain with the paper's stated configuration and report those genuine results.

### C4. Missing Feature Normalization (Z-Score)

**Impact:** Claimed in paper but zero implementation in `main.py`. Unnormalized features destroy model validity and make feature importance meaningless.  
**Fix:** Implement z-score normalization computed on training-set statistics only. Re-train and re-evaluate.

### C5. "5 Independent Runs" Claim Is Unsupported

**Impact:** No multi-seed loop exists in code; standard deviations are invented.  
**Fix:** Implement a multi-seed evaluation loop ($n \geq 10$ seeds, ideally 20+). Save all checkpoints and logs. Report **mean ± standard deviation** for every metric in every table.

### C6. METR-LA Cross-Dataset Evaluation Is Unimplemented

**Impact:** Section 4.8 (~300 words) has no backing code. Results are likely invented.  
**Fix:** Either implement the METR-LA pipeline and report genuine results, or **remove the section entirely**.

---

## 🟠 HIGH (Major Revision Required)

### H1. Mathematical Notation Errors & Ambiguities

**Fix:**

- **Eq. 3.4 (LoRA):** Correct to standard formulation $\mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$ with consistent dimensions.
- **Eq. 3.7 (Attention):** Explicitly define $\mathbf{U}$ shape and state whether attention is computed over time, space, or flattened spatiotemporal grid.
- **Add a comprehensive Notation Table** after problem formulation (~30 symbols).

### H2. Paper–Code Architecture Discrepancies

**Fix:**

- **Feature Fusion (Algorithm 1 vs. model.py:163):** Paper says concatenation $[\mathbf{H}; \mathbf{Z}]$; code uses element-wise addition $\mathbf{H} + \mathbf{Z}$. Align them and justify the choice.
- **Temporal Attention (Section 3.4 vs. model.py:161):** Paper omits mean pooling over time; code applies `mean(dim=2)`. Explicitly document this pooling and justify it.
- **Prediction Head (Section 3.5 vs. model.py:136):** Paper claims 4 horizons; code defaults to 12. Verify `main.py` training script and align.

### H3. Thesis Scaffolding Inappropriate for Journal Article

**Fix:** Remove these sections entirely and weave content into narrative flow:

- 1.1 "Purpose of the Thesis"
- 1.2 "Object and Subject of Research"
- 1.3 "Research Hypotheses"
- 1.4 "Research Questions"
- Appendix "Publications" chapter

### H4. Missing Variance Estimates & Statistical Rigor

**Fix:**

- Re-run all experiments with $n \geq 10$ random seeds.
- Report mean ± SD in **all** tables (Tables 2, 3, 4, 7).
- Add paired t-tests or Wilcoxon signed-rank tests for **every** baseline comparison (not just TCN).
- Report effect sizes (Cohen's $d$) with 95% CIs.
- Apply multiple comparison correction (Holm-Bonferroni).

### H5. Incomplete Ablation Study

**Fix:** Expand Table 3 to include ablations for:

- Number of attention heads ($n_h$)
- Number of graph hops ($K$)
- Auxiliary loss weight ($\lambda$)
- Model dimension ($d_{\text{model}}$)

### H6. District ANOVA Missing Post-Hoc Tests

**Fix:** Add Tukey HSD post-hoc tests to identify which districts differ significantly. Report effect sizes.

### H7. Factual Error in Synthetic Events

**Fix:** "EXPO-2025 Astana" is false (EXPO 2025 is Osaka; Astana hosted EXPO 2017). Replace with a plausible real Astana event or generic "International Exhibition."

### H8. arXiv Preprints Used Where Peer-Reviewed Versions Exist

**Fix:** Replace with journal/conference versions where available:

- `cho2014rnnencoder` → EMNLP 2014
- `bai2018empirical` → TCN peer-reviewed version
- `hendrycks2016gelu`, `ba2016layernorm` → published versions

### H9. Self-Citation Density & Tone

**Fix:** Reduce self-citations in future-work section (3 of 4 bullets cite own work). Cite broader literature for general directions. Soften novelty claims: replace "first ever" with "to the best of our knowledge, no prior work unifies all four simultaneously."

### H10. Complexity Analysis Corrections

**Fix:** Distinguish sparse vs. dense operations:

- Physical graph propagation: $O(K \cdot |\mathcal{E}| \cdot d)$ (not $N^2$)
- Adaptive adjacency: $O(N^2 \cdot d_{\text{emb}})$ (dense, correct as stated)

### H11. Calibration & Feature Attribution Specificity

**Fix:**

- **Calibration:** Consider equal-mass bins instead of equal-width for heavy-tailed count data.
- **Feature Importance:** Specify exact attribution method (e.g., "Integrated Gradients with 50 reference points" or "simple gradients with saturation acknowledgment").

### H12. UI Screenshots in Methodology

**Fix:** Remove all UI screenshots (Figures 3.4–3.8) from main text. If system deployment is a claimed contribution, move to supplementary materials; otherwise delete.

---

## Recommended Execution Order

| Week         | Focus                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Week 1**   | Fix code–paper mismatches (H2, C3, C4). Implement normalization, align hyperparameters, fix LoRA/attention equations. |
| **Week 2**   | Implement real experiments (C2, C5, C6). Add multi-seed loop, retrain all baselines, generate genuine figures.        |
| **Week 3**   | Add real-world dataset or reframe contribution (C1). Expand ablations (H5). Add statistical tests (H4, H6).           |
| **Week 4**   | Structural polish (H3, H12, C6 if removing METR-LA). Fix citations (H8, H9). Correct factual errors (H7).             |
| **Week 5–6** | Global tense consistency, notation table, complexity corrections, final Q1 formatting (~35–40 pages main text).       |

**Bottom line:** Do not submit in current form. The empirical foundation must be rebuilt from genuine experiments before Q1 consideration.

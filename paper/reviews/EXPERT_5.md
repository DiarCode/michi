I will now conduct a thorough, multi-stage peer review of this Master's thesis on DTS-GSSF. Let me begin by carefully parsing the document.

---

## 1. METADATA

- **TITLE:** Real-Time Adaptive Passenger Flow Prediction: A Hybrid Model Approach
- **AUTHORS:** Diar Begisbayev (student), supervised by Aivar Sakhipov
- **VENUE:** Astana IT University, Master's Thesis, June 2026 (institutional degree requirement, not a peer-reviewed venue)

---

## 2. PARSED COMPONENTS (Stage 0)

### PROBLEM:

The thesis addresses multi-horizon passenger flow prediction on the Astana (Kazakhstan) bus network. Given a 72-hour historical context across 374 bus stations with 16 engineered features, the model predicts boarding counts for 4 horizons (15, 30, 60, 120 minutes) using a Negative Binomial distributional output.

### MOTIVATION:

Urban transit in Astana serves 1M+ residents with highly variable demand. Three limitations motivate the work: (L1) convolutional message passing limits fine-grained temporal state transitions; (L2) fixed physical graphs miss latent transfer patterns; (L3) deterministic/Gaussian outputs are inappropriate for overdispersed count data.

### CLAIMED CONTRIBUTIONS:

1. DTS-GSSF architecture unifying gated recurrent encoder, adaptive graph propagation, and multi-head temporal attention
2. Learnable adaptive adjacency matrix complementing physical route topology
3. Negative Binomial likelihood with MSE auxiliary loss for count-data modelling
4. LoRA-based parameter-efficient adaptation for route-level specialization
5. Evaluation on synthetic dataset (1.3M records, 374 stations, 12 months) achieving R² = 0.978 ± 0.001

### METHOD:

- **Gated Recurrent Encoder (GRE):** Single-gate recurrent state update with LoRA-adapted input projection (d_model=192, r=16)
- **GraphPropagation:** K=3 hop message passing with combined physical (symmetric normalized route adjacency) and adaptive (learnable node embeddings E1, E2 with d_emb=16) matrices, mixed via learnable α
- **TemporalAttention:** Multi-head self-attention (n_h=6, d_h=32) over T=72 timesteps, per-station
- **Prediction Heads:** Per-station MLP + aggregate LoRA-adapted linear projection
- **Loss:** Negative Binomial NLL + λ·MSE (λ=0.3)

### EXPERIMENTS:

- **Datasets:** Synthetic Astana (1,296,480 station-hours, 70/15/15 split); LACMTA open benchmark (62,304 windows, 180 stops, 9 routes)
- **Baselines:** HA, Seasonal Naive, MA, LSTM, GRU, TCN, STGCN, Graph WaveNet, AGCRN, ASTGCN (implied but not all reported in Table 5.1)
- **Metrics:** R², MAE, RMSE, MAPE (>5 threshold)
- **Key Results:** R²=0.978±0.001, MAE=11.65±0.14 (all 42 series); R²=0.692 on bottom-28 stations alone; LACMTA R²=0.862

### EXPLICIT LIMITATIONS:

1. Synthetic data may not reflect real-world noise (fare evasion, GPS drift)
2. R²=0.978 on full hierarchy vs. 0.69 on bottom-level suggests irreducible variance
3. O(NT²d) attention complexity may bottleneck for N>2000 or T>200
4. Static graph assumption (routes change due to construction/policy)
5. Real-world validation with Astana APC data remains future work

---

## 3. STRUCTURED SUMMARY (Stage 1)

### SUMMARY/PROBLEM:

The thesis tackles multi-horizon spatiotemporal forecasting of bus passenger boardings in Astana, Kazakhstan. The problem is formulated as predicting Negative Binomial parameters (μ, κ) for each of 374 stations across 4 time horizons, given 72 hours of history with 16 features including weather, calendar, and cyclical encodings.

### SUMMARY/METHOD:

DTS-GSSF processes input through four stages: (1) Gated Recurrent Encoder with LoRA-adapted projection and single-gate state update; (2) GraphPropagation with K=3 hops using a convex combination of physical and adaptive adjacency matrices; (3) TemporalAttention with 6 heads over the full 72-hour context; (4) Dual prediction heads (per-station MLP + aggregate linear) outputting NB parameters. Training uses AdamW with cosine annealing, FP16 mixed precision, and early stopping (patience=50).

### SUMMARY/RESULTS:

On the synthetic Astana dataset, DTS-GSSF achieves R²=0.978±0.001 across all 42 series (28 stations + 14 aggregates), substantially outperforming 10 baselines. Ablation shows graph propagation contributes ΔR²=+0.254 and temporal attention contributes ΔR²=+0.066. On LACMTA open data, R²=0.862. The model maintains R²>0.95 under 30% station dropout and moderate noise. Calibration analysis shows NB outperforms Poisson (ECE comparable to Gaussian, better 50% coverage). Integrated Gradients identifies temperature (0.47) and precipitation (0.20) as dominant features.

### SUMMARY/LIMITATIONS:

The work is primarily evaluated on synthetic data generated from OSM topology and heuristic demand simulation. The high R² on hierarchical aggregates masks lower bottom-level performance (R²=0.692). Complexity may limit scalability to megacities. Graph topology is static. Real-world APC validation is pending.

---

## 4. CRITERION-WISE EVALUATION (Stage 2)

---

### CRITERION: 1) Novelty & Originality

**TEXT-BASED-EVIDENCE:**

- The authors claim DTS-GSSF is "the first model to integrate a gated recurrent temporal encoder with dual adjacency graph propagation, multi-head temporal attention, and Negative Binomial likelihood within a single compact architecture" (Sec. 3.5, Table 3.1)
- The combination of four properties is presented as novel: linear-time temporal encoding, dual graph propagation, long-range temporal attention, and NB likelihood (Sec. 3.5)
- LoRA adaptation is applied to spatiotemporal forecasting, extending prior work on LLM fine-tuning (Sec. 4.3.1, citing Hu et al. [36])
- The GRE is positioned as a simplified alternative to S4/Mamba for moderate context lengths (Sec. 4.3.1)

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: Hybrid architectures combining recurrent/graph/attention components exist in traffic forecasting (e.g., AGCRN combines adaptive GCN with GRU; STGNN variants exist). The specific combination here is a recombination of existing primitives.
- EXTERNAL-KNOWLEDGE: Negative Binomial outputs for count forecasting have been used in DeepAR [31] and Zhu & Laptev [32] (both cited). Application to graph-structured transit data is less common but not unprecedented (Wang et al. [33] cited).
- EXTERNAL-KNOWLEDGE: LoRA for parameter-efficient fine-tuning is well-established in NLP [36]; its application to route-level specialization in transit forecasting is a sensible transfer but not a fundamental methodological novelty.

**STRENGTHS:**

- The dual-adjacency design (physical + adaptive with learnable mixing) is a well-motivated architectural choice that addresses a real limitation of fixed-graph methods
- The hierarchical forecasting structure (28 stations + 14 aggregates) is practically relevant for operational transit planning
- The complete system integration (training + FastAPI backend + React dashboard) shows engineering maturity

**WEAKNESSES:**

- The core components (gated recurrence, graph propagation, temporal attention, NB likelihood) are all well-established primitives; the novelty lies primarily in their combination rather than in fundamental new mechanisms
- The GRE is essentially a simplified GRU with a single gate (no input gate, no output gate, no candidate state gating beyond tanh)—this is actually _less_ expressive than standard GRU/LSTM, not more
- The claim of being "first" is strong given that similar hybrid architectures exist (e.g., AGCRN [9] already combines adaptive graphs with recurrent gating)
- LoRA adaptation for route specialization is practical but technically straightforward; the 94% parameter reduction claim is standard for LoRA with r=16, d=192

**SCORE-[0-10]:** 5

**CONFIDENCE-[0.0-1.0]:** 0.75

---

### CRITERION: 2) Technical Correctness & Soundness

**TEXT-BASED-EVIDENCE:**

- GRE update: s*t = a_t ⊙ s*{t-1} + (1 - a_t) ⊙ b_t where a_t = σ(W_a u_t), b_t = tanh(W_b u_t) (Sec. 4.3.1, Eq. 4.5)
- The GRE lacks an input gate or output gate, making it a single-gate recurrent unit—simpler than GRU
- Adaptive adjacency: A_adp = softmax(ReLU(E1 E2^T)) (Sec. 4.3.2, Eq. 4.7)
- Combined adjacency: A = α·A_phys + (1-α)·A_adp with α = σ(log α_0) (Sec. 4.3.2, Eq. 4.8)
- Graph propagation: h^(k) = GELU(A h^(k-1) W_g) (Sec. 4.3.2, Eq. 4.9)—this is a standard GCN-style propagation
- TemporalAttention operates on reshaped U' ∈ R^(BN)×T×d with multi-head self-attention (Sec. 4.3.3)
- NB NLL formula is standard (Sec. 4.3.5, Eq. 4.12)
- Loss: L = L_NB + λ·L_MSE with λ=0.3 (Sec. 4.3.5, Eq. 4.11)

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: The GRE's single-gate design raises stability concerns. Without an input gate, the state can explode if (1-a_t)⊙b_t has large magnitude. The authors claim "stable gradient propagation" but provide no theoretical or empirical evidence (e.g., gradient norm analysis) beyond training convergence curves.
- EXTERNAL-KNOWLEDGE: The graph propagation in Eq. 4.9 uses the same weight matrix W_g at every hop—this is standard but limits expressivity compared to hop-specific weights. No mention of whether A is normalized per hop, which affects numerical stability.
- EXTERNAL-KNOWLEDGE: The NB likelihood uses a global κ shared across all stations and horizons. This assumes homogeneous dispersion, which may not hold (central business districts likely have different overdispersion than peripheral residential areas).

**STRENGTHS:**

- The problem formulation is mathematically clean and well-defined
- The NB likelihood is theoretically appropriate for count data and overdispersion
- The dual-adjacency mixing with learnable α is a principled way to balance prior structure and data-driven discovery
- Gradient clipping (max norm 1.0) and cosine annealing are sensible training choices

**WEAKNESSES:**

- The GRE is technically a _degraded_ GRU (missing input/output gates), yet the authors claim it provides "more stable gradient propagation" without evidence—this is an unsupported claim (Sec. 4.3.1: "the single-gate design provides stable gradient propagation over 72 steps without the numerical instability observed in deeper LSTM stacks"—no citation or ablation supports this)
- The adaptive adjacency uses softmax over ReLU(E1 E2^T), which can produce dense attention even for distant stations. No sparsity constraint or locality bias is applied, potentially allowing unrealistic long-range dependencies
- The NB dispersion κ is global (shared across all stations/horizons), which is a strong and likely incorrect assumption. The ablation study does not test station-specific or horizon-specific κ
- The MSE auxiliary loss (λ=0.3) is added to NLL, but MSE on count data with NB parameterization is unusual—typically one would use a Gaussian NLL or stick to pure NB. The interaction between MSE and NB NLL is not analyzed
- No theorem, proof, or formal analysis of convergence, expressivity, or approximation guarantees is provided

**SCORE-[0-10]:** 5

**CONFIDENCE-[0.0-1.0]:** 0.70

---

### CRITERION: 3) Methodological Rigor & Experimental Design

**TEXT-BASED-EVIDENCE:**

- Dataset: Synthetic, 1,296,480 station-hours, Jan-Dec 2025, 374 stations, 10 routes (Sec. 4.2.2, Table 4.2)
- Train/Val/Test: 70%/15%/15% split (Table 4.2)
- DTS-GSSF evaluated over 3 seeds; baselines: single-seed (Table 5.1 caption)
- Ablation study: 5 variants (v1-v5) on single-seed (Table 5.3)
- Hyperparameter search space reported (Table A.4)
- LACMTA open dataset used for cross-validation (Sec. 4.2.3, Table 5.2)
- Robustness: station dropout, Gaussian noise, contiguous gaps (Table 5.7)
- Calibration: NB vs Gaussian vs Poisson (Table 5.5)
- Feature importance: Integrated Gradients (Fig. 5.8)
- Per-district analysis: 4 districts (Table 5.6)

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: 3 seeds for the main model is minimal for Q1-level work; 5-10 seeds is standard. The baselines are single-seed, making statistical comparison impossible (no variance estimate for baselines).
- EXTERNAL-KNOWLEDGE: The synthetic data generation process uses heuristic multipliers (weather, holidays, events) with no validation against real ridership distributions. The log-normal base demand with sinusoidal patterns is a common but simplified model that may not capture real transit demand complexity (e.g., burstiness, route interactions, mode competition).

**STRENGTHS:**

- Comprehensive evaluation across multiple dimensions: main results, ablation, per-horizon, calibration, robustness, feature importance, per-district, cross-dataset
- The open LACMTA benchmark provides some external validation
- Multiple perturbation scenarios test model robustness
- Calibration analysis is included (often missing in forecasting papers)

**WEAKNESSES:**

- **Critical:** The primary evaluation is on synthetic data. The authors acknowledge this but it fundamentally limits the credibility of the R²=0.978 claim. Real transit data contains complexities (fare evasion, GPS errors, schedule adherence, special events, competition from ride-hailing) that synthetic generators typically miss
- The baselines run on single seed while DTS-GSSF runs on 3 seeds—this is an unfair comparison. The large gap (R²=0.978 vs 0.718 for AGCRN) may be inflated by this mismatch, especially since AGCRN on bottom-28 stations is compared to DTS-GSSF on all 42 series
- The ablation study is single-seed, making the ΔR² claims (e.g., +0.254 for graph) unreliable. No confidence intervals on ablation contributions
- The "10 baselines" claim includes classical methods (HA, Seasonal Naive, MA) that are strawmen for this task—they are expected to perform poorly and their inclusion inflates the apparent improvement
- No statistical significance test (t-test, Wilcoxon) is reported for the main comparison—only for the LACMTA result is a Wilcoxon test mentioned (but this appears to reference prior work [14], not the current study)
- The LACMTA dataset is described as "publicly available" but no download link or specific dataset identifier is provided, hindering reproducibility
- Hyperparameter search space is small (3 values per parameter) and no mention of search strategy (grid vs random vs Bayesian)

**SCORE-[0-10]:** 4

**CONFIDENCE-[0.0-1.0]:** 0.80

---

### CRITERION: 4) Empirical Results & Analysis

**TEXT-BASED-EVIDENCE:**

- Main result: R²=0.978±0.001, MAE=11.65±0.14 (Table 5.1)
- Bottom-level only: R²=0.692 (Sec. 5.2)
- Ablation: v1 (no graph, no TA): R²=0.724; v2 (no TA): R²=0.912; v3 (full): R²=0.978 (Table 5.3)
- LACMTA: R²=0.862, MAE=7.1 (Table 5.2)
- Per-horizon: R² stable at ~0.978 across all 4 horizons (Sec. 5.5)
- Calibration: NB ECE=0.291, 50% cov=0.723, 90% cov=0.972 (Table 5.5)
- Feature importance: temperature=0.47, precipitation=0.20 (Fig. 5.8)
- Robustness: 30% dropout → R²=0.951; σ=0.5 noise → R²=0.958 (Table 5.7)

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: The R²=0.978 on synthetic data with known generative process is suspiciously high. In real-world transit forecasting, R²>0.9 is rare due to inherent unpredictability (special events, accidents, weather extremes). The synthetic generator may have made the task easier than reality.
- EXTERNAL-KNOWLEDGE: The bottom-level R²=0.692 is more realistic and comparable to baseline performance (LSTM/GRU/TCN at ~0.697). This suggests the 0.978 figure is largely driven by hierarchical aggregates, which are easier to predict (averaging reduces variance).

**STRENGTHS:**

- The hierarchical forecasting structure is practically useful and the aggregate predictions are accurate
- Cross-dataset validation on LACMTA shows generalization (R²=0.862)
- Robustness analysis demonstrates graceful degradation
- Feature importance analysis provides actionable insights (weather dominance)

**WEAKNESSES:**

- The headline R²=0.978 is misleading because it includes 14 aggregate series that are mathematically easier to predict. The bottom-level R²=0.692 is only marginally better than LSTM/GRU/TCN (0.696-0.697), suggesting the spatial and hierarchical components provide limited benefit at the station level where operational decisions are made
- The ablation ΔR² values are reported without confidence intervals (single-seed), making it impossible to assess whether +0.066 for temporal attention is statistically meaningful
- The LACMTA result (R²=0.862) is better than baselines but the gap is smaller than on synthetic data, suggesting some overfitting to the synthetic generator's patterns
- The calibration analysis shows NB and Gaussian have nearly identical ECE (0.290 vs 0.291), undermining the claim that NB provides substantially better calibration. The 50% coverage improvement (0.723 vs 0.678) is modest
- No analysis of failure modes: when does the model fail? What types of stations/events cause large errors?

**SCORE-[0-10]:** 4

**CONFIDENCE-[0.0-1.0]:** 0.80

---

### CRITERION: 5) Clarity & Organization

**TEXT-BASED-EVIDENCE:**

- Well-structured thesis format with standard chapters (Introduction, Literature Review, Methodology, Results, Discussion, Conclusion)
- Table 4.1 provides clear notation summary
- Figure 4.1 shows architecture diagram (though images are omitted in text)
- Algorithm 1 presents forward pass pseudocode
- Table 4.3 lists all hyperparameters
- Multiple tables and figures support claims

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: The writing is generally clear but occasionally verbose. Some sections (e.g., complexity analysis in 4.4.3) are well-done; others (e.g., GRE relation to SSMs in 4.3.1) are somewhat hand-wavy.

**STRENGTHS:**

- Notation is consistent and well-defined
- Architecture is described at multiple levels (high-level Fig. 4.1, component details, algorithmic pseudocode)
- Hyperparameters and training details are comprehensive (Table 4.3)
- The system interface description (Sec. 4.5) adds practical context

**WEAKNESSES:**

- Several figures are referenced but not visible in the text (marked as "picture intentionally omitted"), making it impossible to verify claims about visualizations
- The distinction between "Gated Recurrent Encoder" and standard GRU is unclear—the GRE appears to be a simplified GRU yet is presented as novel
- The relationship between the 5 publications in Chapter 2 and the current work is somewhat repetitive; the thesis could be tightened by integrating this into the literature review
- Some claims are unsupported by cross-references (e.g., "preliminary experiments showed that GRE dropout harms the model's ability"—no citation to these experiments)
- The code release promise ("upon acceptance") is vague for a thesis, which typically doesn't have an "acceptance" process in the same sense as a journal

**SCORE-[0-10]:** 6

**CONFIDENCE-[0.0-1.0]:** 0.85

---

### CRITERION: 6) Reproducibility & Openness (code/data)

**TEXT-BASED-EVIDENCE:**

- Hardware and software environment specified (Appendix A.5)
- PyTorch 2.6.0, CUDA 12.6, specific package versions listed
- Hyperparameter search space and final values reported (Table A.4)
- "The complete source code, training scripts, and model weights will be released upon acceptance" (Sec. A.5)
- Synthetic dataset generation process described in detail (Sec. 4.2.2)
- LACMTA dataset mentioned but no specific identifier provided

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: The "upon acceptance" promise is problematic for a thesis, which is already a completed work. For Q1 venues, code/data availability at submission is increasingly expected (e.g., NeurIPS/ICML reproducibility standards).

**STRENGTHS:**

- Detailed environment specification aids reproducibility
- Synthetic data generation is described with sufficient detail to reimplement
- Hyperparameter choices are justified with ablation references

**WEAKNESSES:**

- **Critical:** No code, data, or model weights are provided at the time of evaluation. The "upon acceptance" promise is insufficient for verification
- The LACMTA dataset lacks a specific URL or identifier, making the open-data claim unverifiable
- No mention of random seeds used, beyond "3 independent seeds"
- The synthetic data generator code is not provided, and the heuristic parameters (e.g., weather multipliers, event attendance ranges) are not fully specified—only summary descriptions are given
- No training logs, loss curves (beyond Fig. 5.6), or intermediate checkpoints are shared

**SCORE-[0-10]:** 3

**CONFIDENCE-[0.0-1.0]:** 0.90

---

### CRITERION: 7) Significance & Impact for the Field

**TEXT-BASED-EVIDENCE:**

- The work targets real-time transit intelligence platforms (Abstract, Sec. 7.1)
- Claims practical applicability: inference <5ms, 2.2MB model size, edge deployment potential
- Proposes integration with automated data-preparation framework [16]
- Identifies future directions including multi-modal networks, dynamic graphs, causal intervention, edge deployment

**EXTERNAL-KNOWLEDGE:**

- EXTERNAL-KNOWLEDGE: Transit demand forecasting is a well-studied problem with established benchmarks (e.g., PeMS, METR-LA, NYC taxi/bike). The Astana-specific focus limits general impact unless the methodology transfers broadly.
- EXTERNAL-KNOWLEDGE: The hierarchical forecasting contribution (28 stations + 14 aggregates) is practically useful for transit operators but not a major technical advance—hierarchical forecasting has extensive literature (e.g., Hyndman's work on coherent forecasts).

**STRENGTHS:**

- The system integration (dashboard, FastAPI backend, WebSocket alerts) shows awareness of operational requirements
- LoRA adaptation for route specialization is a practical contribution for deployment scenarios
- The focus on a mid-sized city (Astana) addresses a gap in literature dominated by megacity benchmarks

**WEAKNESSES:**

- The synthetic data evaluation fundamentally limits practical impact—transit operators cannot act on a model that has not been validated on real data
- The R²=0.978 headline may mislead practitioners about achievable performance on real systems
- No collaboration with Astana transit authority is reported, and no real deployment or pilot study is described
- The "practical implications" section (Sec. 7.1) is speculative ("can be updated within approximately 2 seconds", "dispatchers can distinguish between routine demand fluctuations") without operational validation
- The field impact is limited by the lack of open benchmarks or code that others can build upon

**SCORE-[0-10]:** 4

**CONFIDENCE-[0.0-1.0]:** 0.75

---

## 5. SELF-CRITICISM / VERIFICATION (Stage 3)

### SELF-CRITIQUE:

Upon re-reading, I identified the following issues in my initial evaluation:

1. **Overstated novelty concern:** I initially scored novelty at 5, but I should verify whether the specific combination (GRE + dual adjacency + temporal attention + NB) has indeed appeared before. The authors' claim in Sec. 3.5 is specific. However, I cannot verify this from the provided text alone, and my external knowledge suggests similar combinations exist. I will maintain the score but note this uncertainty.

2. **Potential misreading of bottom-level performance:** The text states "On bottom-level stations alone, DTS-GSSF achieves R²=0.692, comparable to the baselines." I initially interpreted this as a weakness, but the authors present it transparently. However, the comparison is still unfair because baselines run on single seed.

3. **GRE stability claim:** My criticism that the GRE stability claim is unsupported is valid—the text states "without the numerical instability observed in deeper LSTM stacks" but provides no evidence (no gradient norm plots, no instability comparison). This remains a weakness.

4. **LACMTA statistical test:** I noted that a Wilcoxon test is mentioned for LACMTA but this appears to reference prior work [14], not the current study. Re-checking: Table 5.2 shows single-value results for LACMTA with no variance or statistical test. The Wilcoxon test (p=0.018) is from publication [14] (X-FedFormer), not the current DTS-GSSF evaluation.

### VERIFICATION-QA:

**Q1: Does the paper clearly specify the random seeds used for the 3-seed evaluation?**
A1: NOT SPECIFIED. The text mentions "3 independent seeds" and "3 seeds" but does not list the actual seed values.

**Q2: Are the chosen baselines appropriate and fairly compared?**
A2: PARTIALLY. The baselines include appropriate methods (LSTM, GRU, TCN, STGCN, Graph WaveNet, AGCRN) but the comparison is unfair because baselines use single seed while DTS-GSSF uses 3 seeds. Also, classical baselines (HA, Seasonal Naive, MA) are strawmen.

**Q3: Is there enough ablation to support the claim that graph propagation contributes ΔR²=+0.254?**
A3: NO. The ablation is single-seed with no confidence intervals. The large ΔR² could be due to random variation. Moreover, v1 (no graph, no TA) removes two components simultaneously, confounding their individual contributions.

**Q4: Does the paper provide evidence for the claim that GRE provides "more stable gradient propagation" than LSTM/GRU?**
A4: NO. The claim in Sec. 4.3.1 is unsupported by gradient norm analysis, training curves comparing stability, or citation. Only convergence curves (Fig. 5.6) are shown, but these show final performance, not stability during training.

**Q5: Is the NB likelihood's superiority over Gaussian properly supported?**
A5: PARTIALLY. Table 5.5 shows NB achieves 50% coverage of 0.723 vs Gaussian 0.678, but ECE is nearly identical (0.291 vs 0.290). The claim of "better calibration" is weakly supported.

**Q6: Does the paper specify how the synthetic data weather templates were constructed?**
A6: PARTIALLY. Sec. 4.2.4 mentions "eight templates (clear, cloudy, rain, snow, blizzard, fog, extreme cold, heatwave) with seasonally adjusted sampling probabilities" but does not provide the actual probabilities or template parameters.

**Q7: Is the LACMTA dataset identifier provided for reproducibility?**
A7: NO. The text describes the dataset but provides no URL, API endpoint, or specific dataset name beyond "LACMTA Open Data Portal."

### ADJUSTMENTS:

- No score adjustments made. The verification confirms my initial concerns about methodological rigor (single-seed ablations, unfair baseline comparison, unsupported stability claims) and reproducibility (no code/data, vague LACMTA reference).

---

## 6. FINAL RECOMMENDATION (Stage 4)

**OVERALL-SCORE-[0-10]:** 4

**OVERALL-CONFIDENCE-[0.0-1.0]:** 0.75

**RECOMMENDATION:** Weak Reject

**ONE-PARAGRAPH-JUSTIFICATION:**
This Master's thesis presents DTS-GSSF, a hybrid architecture for transit demand forecasting with sensible design choices (dual adjacency, NB likelihood, hierarchical structure). However, the work suffers from critical methodological limitations that preclude Q1 acceptance. The primary evaluation relies on synthetic data with a known generative process, producing an unrealistically high R²=0.978 that masks bottom-level performance (R²=0.692) only marginally better than simple baselines. The experimental design is flawed: baselines run on single seeds while the proposed method uses three, and the ablation study is single-seed without confidence intervals, making component contribution claims unreliable. Key technical claims—such as the GRE's superior stability over LSTM/GRU—lack supporting evidence. Reproducibility is severely limited by the absence of released code, data, or model weights. While the system integration and practical focus are commendable, the synthetic-only evaluation, unfair comparisons, and unsupported claims prevent this work from meeting Q1 standards. For journal publication, real-world validation, rigorous statistical testing, and artifact release are essential prerequisites.

**AUTHOR-FACING-COMMENTS:**

**Major Strengths:**

1. The dual-adjacency design (physical + adaptive with learnable mixing) is well-motivated and addresses a genuine limitation of fixed-graph methods in transit forecasting.
2. The hierarchical forecasting structure (station-level + aggregate) is practically relevant for operational planning.
3. The system integration (FastAPI backend, React dashboard, WebSocket alerts) demonstrates awareness of deployment requirements.
4. The inclusion of calibration analysis and robustness testing shows methodological breadth.

**Major Weaknesses:**

1. **Synthetic data evaluation:** The R²=0.978 headline is misleading because (a) it includes aggregate series that are easier to predict, and (b) synthetic data with known generation dynamics does not reflect real-world complexity. You must validate on real Automatic Passenger Counting (APC) data from Astana before claiming practical applicability.
2. **Unfair baseline comparison:** Running baselines on single seeds while your method uses three seeds inflates apparent improvements. All methods should use matched seeds (≥5) with reported means and standard deviations.
3. **Unsupported stability claims:** The claim that GRE provides "more stable gradient propagation" than LSTM/GRU (Sec. 4.3.1) requires evidence—provide gradient norm traces across training epochs or training loss curves comparing stability.
4. **Single-seed ablations:** The ΔR²=+0.254 (graph) and +0.066 (attention) claims are unreliable without multiple seeds. Run each ablation with ≥5 seeds and report confidence intervals.
5. **Global dispersion assumption:** The NB likelihood uses a single global κ. Test station-specific or district-specific κ—peripheral residential areas likely have different overdispersion than central business districts.

**Concrete Suggestions for Improvement:**

1. **Real-world validation:** Partner with Astana's transit authority to obtain APC data. Report performance on real data alongside synthetic results. Expect significantly lower R² on real data.
2. **Fair comparison:** Re-run all baselines with 5-10 seeds, matching your experimental protocol. Report statistical significance (paired t-test or Wilcoxon signed-rank) for all main comparisons.
3. **Ablation rigor:** Run ablations with multiple seeds. Consider a more granular ablation that separates graph propagation and temporal attention (v1 removes both, making their individual contributions confounded).
4. **Release artifacts:** Publish code, model weights, and the synthetic dataset generator on GitHub or Zenodo. For the LACMTA benchmark, provide exact dataset identifiers and preprocessing scripts.
5. **Failure analysis:** Add a section analyzing when/where the model fails. Which stations? Which weather conditions? Which event types? This is essential for operational deployment.
6. **Theoretical grounding:** If claiming GRE stability, provide a simple analysis (e.g., bound on gradient norms) or empirical evidence (gradient histograms). Consider relating the GRE more formally to SSM theory.
7. **Dynamic graphs:** Address the static graph limitation (Sec. 6.2) with at least a conceptual design or preliminary experiment for time-varying adjacency.

**Minor Issues:**

- Fix the "upon acceptance" language for a thesis (theses are not "accepted" in the journal sense).
- Provide exact LACMTA dataset URLs/API endpoints.
- Clarify whether ASTGCN [10] was actually evaluated—Table 5.1 does not include it despite being listed as a baseline in Sec. 5.2.
- The GRE is essentially a single-gate RNN—consider whether this simplification is truly advantageous or merely a design choice that happened to work on your synthetic data.

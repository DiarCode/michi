# Deep Peer Review & Q1 Readiness Assessment
## Master Thesis: "Real-Time Adaptive Passenger Flow Prediction: A Hybrid Model Approach"
**Date:** 2026-05-28  
**Reviewer:** Dr. Alexandra Mercer (Senior Research Scientist, Q1 ML/AI Editorial Consultant)  
**Target Standard:** Q1 Scopus-indexed journal (ML/AI or Transportation Informatics)  

---

## 1. Executive Summary

This document provides a comprehensive, section-by-section critique of the Master's thesis submitted by Diar Begisbayev (Astana IT University). The underlying research---DTS-GSSF (Dual-Timescale Graph Gated Forecasting)---exhibits genuine architectural novelty, strong empirical methodology, and commendable reproducibility discipline. However, **the manuscript in its current form is not yet ready for submission to a Q1 Scopus-indexed journal**.

The three most critical gaps are:
1. **Absence of real-world ridership data validation.** The entire empirical contribution rests on a synthetic dataset. No Q1 venue in traffic forecasting or spatiotemporal ML will accept a primary contribution that has not been validated on at least one real-world benchmark.
2. **Insufficient statistical reporting.** Main results are reported as point estimates without standard deviations, confidence intervals, or rigorous multiple-comparison correction.
3. **Structural bloat from thesis-to-paper conversion.** The manuscript retains bureaucratic thesis scaffolding ("Purpose of the Thesis," "Object and Subject of Research," UI screenshots, a full "Publications" chapter) that violates the conciseness norms of Q1 ML journals.

The good news: these are solvable problems. With targeted revision---real-data validation, statistical tightening, mathematical polishing, and aggressive condensation---this work can reach Q1 competitiveness within 4--6 weeks of focused effort.

---

## 2. Critical Weaknesses (Blocking Q1 Publication)

These issues must be resolved before any submission to a Q1 venue. A reviewer would likely recommend **Reject** or **Major Revision** on the basis of these alone.

### 2.1. Synthetic-Only Primary Evaluation
**Severity:** Critical  
**Location:** Chapter 4 (Results), Section 3.2 (Dataset)  
**Finding:** The model is trained and evaluated exclusively on a synthetic dataset of 1.3M records generated from OpenStreetMap topology and heuristic demand simulation. While the simulation is sophisticated, a Q1 journal requires empirical evidence that the model generalizes to real-world data.

**Why this is a blocker:**  
- Reviewers in *IEEE Transactions on Intelligent Transportation Systems*, *Transportation Research Part C*, or *Neural Networks* will immediately flag this as the primary weakness.
- The METR-LA cross-dataset evaluation (Section 4.8) evaluates **traffic speed** (continuous, Gaussian head), not ridership counts. This does not validate the core Negative-Binomial forecasting claim.
- No comparison between synthetic statistics and a small real sample is provided (e.g., mean/variance alignment, distributional match).

**Required Action:**  
- Obtain a real-world ridership dataset, even if small or from another city. Public benchmarks: PeMS (California), NYC MTA turnstile data, or the Hangzhou Metro dataset (used in STGNN literature).
- If real data is truly unavailable, the paper must be reframed as a **methodology/algorithmic contribution** with extensive synthetic ablation and theoretical analysis, and submitted to a theory-friendly venue---but this is a weaker path.
- At minimum, provide a statistical validation that the synthetic data matches known properties of real transit demand (e.g., overdispersion parameters from published studies).

### 2.2. Missing Variance Estimates in Results
**Severity:** Critical  
**Location:** Tables 2, 3, 4, 7  
**Finding:** All main result tables report single point estimates (e.g., $R^2 = 0.889$). The text mentions "5 independent runs," but standard deviations, confidence intervals, or p-values are not tabulated.

**Why this is a blocker:**  
- Q1 ML journals require reproducible statistics. A single run is anecdote, not evidence.
- The paired t-test is mentioned only for DTS-GSSF vs. TCN. What about vs. AGCRN, Graph WaveNet, etc.?
- 5 runs is the absolute minimum; 10 runs is standard, and 20+ is preferred for top-tier venues.

**Required Action:**  
- Re-run all experiments with $n \geq 10$ random seeds.
- Report mean $\pm$ standard deviation for all metrics.
- Add a table of p-values (or better, effect sizes with 95% CIs) for all pairwise comparisons against the strongest baseline.
- Apply a multiple comparison correction (e.g., Holm-Bonferroni) if reporting many p-values.

### 2.3. Mathematical Notation Errors and Ambiguities
**Severity:** High  
**Location:** Section 3.3 (Eq. 3.4), Section 3.4 (Eq. 3.7), Algorithm 1  
**Finding:**

**Eq. 3.4 (LoRA projection):**  
```latex
\mathbf{W}_{\text{in}}(\mathbf{x}) = \mathbf{W}_{\text{base}} \mathbf{x} + \frac{\alpha}{r} (\mathbf{x} \mathbf{A}^\top) \mathbf{B}^\top
```
This is dimensionally inconsistent if $\mathbf{x}$ is a column vector ($F \times 1$). The standard LoRA formulation is $\mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x}$ where $\mathbf{B} \in \mathbb{R}^{d_{out} \times r}, \mathbf{A} \in \mathbb{R}^{r \times d_{in}}$.

**Eq. 3.7 (Attention):**  
The text states $\mathbf{U} \in \mathbb{R}^{T \times d_{\text{model}}}$ is "the tensor of per-timestep GRE outputs reshaped across stations." This is ambiguous. Is the attention computed independently per station (time-axis self-attention) or across stations (spatial self-attention)? If per station, $\mathbf{U}$ should be indexed by station: $\mathbf{U}_i \in \mathbb{R}^{T \times d}$. If reshaped across all stations, $\mathbf{U} \in \mathbb{R}^{T \times (N \cdot d)}$ or $\mathbb{R}^{(N \cdot T) \times d}$, which changes complexity and interpretation entirely.

**Required Action:**  
- Fix the LoRA equation to match the standard formulation with consistent dimensions.
- Redraw or rewrite the TemporalAttention description to explicitly state: (a) the shape of $\mathbf{U}$, (b) over which dimension(s) self-attention is computed, and (c) whether the operation is parallelized across stations.
- Add a comprehensive **Notation Table** (see Section 5.1).

### 2.4. Thesis Scaffolding Inappropriate for Journal Article
**Severity:** High  
**Location:** Chapter 1 (Introduction)  
**Finding:** The Introduction contains four thesis-specific bureaucratic sections that do not exist in Q1 journal articles:
- 1.1 "Purpose of the Thesis"
- 1.2 "Object and Subject of Research"
- 1.3 "Research Hypotheses"
- 1.4 "Research Questions"

**Why this is a blocker:**  
- Q1 journals expect IMRaD structure. Hypotheses and RQs can be woven into the Introduction's narrative flow, not enumerated as separate subsections.
- The "Purpose" paragraph is a single 70-word run-on sentence that is nearly unreadable.
- These sections push the actual scientific argument (contributions) below the fold.

**Required Action:**  
- Merge hypotheses into the Introduction's problem formalization paragraph.
- Merge RQs into the contribution statement or experimental setup.
- Delete "Purpose," "Object," and "Subject" entirely. They add zero scientific content.

---

## 3. Structural & Organizational Review

### 3.1. Chapter Architecture

| Current Name | Q1 Standard | Verdict |
|---|---|---|
| Chapter 1: Introduction | Introduction | Acceptable, but remove thesis scaffolding |
| Chapter 2: Literature Review | Related Work | Rename to "Related Work"; condense by 30% |
| Chapter 3: Methodology | Method / Proposed Approach | Rename; split dataset and system UI |
| Chapter 4: Results | Experiments / Results | Acceptable; add variance columns |
| Chapter 5: Discussion | Discussion | Acceptable; strengthen limitation specificity |
| Chapter 6: Conclusion | Conclusion | Acceptable; shorten practical implications |
| Appendix: Publications | --- | **Remove.** This is a thesis artifact. |

**Key Recommendation:** For journal submission, the paper should be condensed to **~35--40 pages of main text** (including figures). The current thesis is likely 60+ pages.

### 3.2. Missing "Background and Preliminaries" Section
**Severity:** Medium  
**Location:** Between Chapters 2 and 3  
**Finding:** The paper jumps directly from Related Work to Methodology without a formal problem definition and notation table. While Section 3.1 provides problem formulation, it is buried inside the Methodology chapter.

**Required Action:**  
- Add a short "Preliminaries" section (or subsection) immediately after Related Work that defines:
  - The graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$
  - The input tensor $\mathbf{X}$ and output parameters $(\mu, \kappa)$
  - A **Notation Table** listing all symbols, dimensions, and meanings

### 3.3. UI Screenshots in Methodology
**Severity:** Medium  
**Location:** Section 3.6, Figures 3.4--3.8  
**Finding:** Five full-page UI screenshots (React dashboard, Streamlit interfaces) consume ~5 pages in the methodology chapter.

**Why this hurts Q1 readiness:**  
- Q1 ML journals evaluate scientific contribution, not software engineering.
- These figures dilute the technical density of the methodology.
- Captions describe UI functionality ("The command centre dashboard serves as the primary monitoring interface...") rather than scientific insight.

**Required Action:**  
- Remove all UI screenshots from the main text.
- If system deployment is a claimed contribution (it is not explicitly listed in the contribution bullets), move the description to a single paragraph and the figures to a supplementary materials appendix.

---

## 4. Language, Style & Register

### 4.1. Tense Inconsistencies
**Severity:** Medium  
**Location:** Global  
**Finding:** The manuscript frequently mixes tenses inappropriately.

**Examples:**
- **Abstract:** "We present DTS-GSSF..." (present) mixed with "DTS-GSSF achieves..." (present) and "Ablation studies confirm..." (present). For a structured abstract, past tense should dominate: "We proposed... evaluated... achieved..."
- **Introduction, line 11:** "We address the problem..." (present) --- should be "We addressed" or "This paper addresses."
- **Results, line 47:** "DTS-GSSF achieves the highest $R^2$..." --- past tense is required for empirical results: "DTS-GSSF achieved..."
- **Conclusion:** "We presented DTS-GSSF..." (correct, past) but "Our key findings are threefold" (present) --- both are acceptable in conclusion, but consistency within paragraphs is lacking.

**Required Action:**  
- Apply a global pass enforcing the following rules:
  - **Past tense:** All experiment descriptions, result reporting, and related work summaries.
  - **Present tense:** General facts, mathematical definitions, paper structure references, and contribution statements (as enduring truths).
- Example fix for Abstract:
  > "We proposed DTS-GSSF... The model integrated... We evaluated... DTS-GSSF achieved..."

### 4.2. Forbidden Phrases & Academic Red Flags
**Severity:** Medium  
**Location:** Global  
**Finding:** Several phrases flagged by Q1 editorial standards appear:

1. **"The proposed method"** (Conclusion, line 1; Discussion, line 9).  
   **Fix:** Replace with "DTS-GSSF" or "our method."

2. **"To our knowledge, the first model to integrate..."** (Literature Review, line 110).  
   **Fix:** This is a dangerous novelty claim. Replace with: "DTS-GSSF unifies these four properties within a single compact architecture. To the best of our knowledge, no prior work combines all four simultaneously."

3. **"State-of-the-art results"** (not explicitly present, but the $R^2$ ceiling discussion implies it).  
   **Fix:** Always qualify: "...among methods evaluated on the synthetic Astana benchmark."

4. **"It is worth noting that..."** (not present, good). Maintain avoidance.

5. **"Obviously" / "Clearly" / "Trivially"** (not present, good). Maintain avoidance.

6. **"Future work will explore..."** (Conclusion, line 40).  
   **Fix:** "A promising direction is to extend..." (specific, actionable).

### 4.3. Precision and Quantitative Language
**Severity:** Medium  
**Location:** Results and Discussion  
**Finding:** Some comparisons lack baseline specificity.

**Examples:**
- "Outperforming baseline approaches including Historical Average, LSTM, GRU, and TCN" (Abstract).  
   **Fix:** "...outperforming the strongest baseline (AGCRN) by 2.0\% absolute $R^2$ and 6.9\% relative MAE."
- "The model converges within 90 epochs" (Abstract).  
   **Fix:** "Training converged in a mean of 90 epochs (SD = 12) across 10 independent runs."
- "Computational cheap" (Literature Review, line 14).  
   **Fix:** "Computationally inexpensive."

### 4.4. Redundancy and Filler
**Severity:** Low-Medium  
**Location:** Introduction, Methodology  
**Finding:**
- **Self-referential meta-commentary:** "A detailed account of the author's peer-reviewed publications... is provided in Chapter..." (Introduction, line 65). Delete.
- **Over-explanation of well-known concepts:** The LoRA description in Section 3.3 includes a full paragraph explaining what LoRA is. A Q1 audience knows LoRA; cite and move on.
- **Hyperparameter justification paragraph** (end of Section 3.5) is a 200-word wall of text. Convert to a bullet list or move to appendix.

---

## 5. Mathematical & Technical Rigor

### 5.1. Notation Table (Missing)
**Severity:** High  
**Finding:** The paper introduces approximately 30 distinct mathematical symbols across Sections 3.1--3.5. No notation table exists.

**Required Action:**  
Add a table immediately after the problem formulation:

| Symbol | Type | Dimension | Meaning |
|---|---|---|---|
| $\mathcal{G}$ | Graph | --- | Bus network graph |
| $N$ | Scalar | --- | Number of stations (374) |
| $T$ | Scalar | --- | Context window in hours (72) |
| $\mathbf{X}$ | Tensor | $\mathbb{R}^{B \times T \times N \times F}$ | Input feature tensor |
| $\mathbf{s}_t$ | Matrix | $\mathbb{R}^{N \times d}$ | GRE state at time $t$ |
| $\mathbf{A}_{\text{phys}}$ | Matrix | $\mathbb{R}^{N \times N}$ | Normalized physical adjacency |
| $\mathbf{A}_{\text{adp}}$ | Matrix | $\mathbb{R}^{N \times N}$ | Learned adaptive adjacency |
| $\alpha$ | Scalar | $[0,1]$ | Adjacency mixing coefficient |
| $\mu_{h,i}$ | Scalar | --- | Predicted mean for station $i$, horizon $h$ |
| $\kappa$ | Scalar | --- | Global NB dispersion parameter |
| $\lambda$ | Scalar | --- | Auxiliary MSE loss weight (0.3) |

### 5.2. Equation-Specific Issues

**Eq. 2.1 (GCN propagation):**  
The equation uses $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$. This is the symmetric normalization. The text says "symmetric-normalised propagation rule" --- correct.

**Eq. 2.3 (Linear SSM):**  
The standard linear SSM uses $\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \mathbf{B} \mathbf{x}_t$. The equation is correct. However, the text does not define the dimensions of $\mathbf{A}, \mathbf{B}, \mathbf{C}$. Add dimensions.

**Eq. 3.1 (Prediction):**  
Uses $\hat{y}_{t+h,i} \sim \text{NB}(\mu, \kappa)$. The symbol $\hat{y}$ conventionally denotes a point estimate, not a random variable. Use $Y_{t+h,i}$ for the random variable and $\hat{y}$ for the point prediction.

**Eq. 3.2 (Base demand):**  
$b_{t,i} \sim \text{LogNormal}(\mu_{\text{base},i} + f_{\text{hour}}(t) + f_{\text{dow}}(t), \sigma_{\text{base}}^2)$.  
The notation $f_{\text{hour}}(t)$ and $f_{\text{dow}}(t)$ is introduced without definition. Are these learned embeddings? Fixed sinusoidal encodings? Define them.

**Eq. 3.3 (Demand modulation):**  
$y_{t,i} = \lfloor b_{t,i} \cdot w_{\text{temp}} \cdot w_{\text{precip}} \cdot w_{\text{holiday}} \cdot w_{\text{event}} \rfloor$.  
The floor function ensures integrality, but the LogNormal base demand already produces continuous values. Explain why the multiplicative heuristic is chosen over a generative model (e.g., compound Poisson).

**Eq. 3.6 (GRE update):**  
$\mathbf{s}_t = \mathbf{a}_t \odot \mathbf{s}_{t-1} + (1 - \mathbf{a}_t) \odot \mathbf{b}_t$.  
This is essentially a single-gate GRU (or an LSTM without input/output gates). The text calls it a "Gated Recurrent Encoder" and compares it to S4/Mamba, but mathematically it is much simpler. Be honest about the relationship: "The GRE employs a single multiplicative forget gate, resembling a minimal gated unit [CITE]."

**Eq. 3.9 (Loss):**  
$\mathcal{L} = \text{NLL}(y, \mu, \kappa) + \lambda \, \text{MSE}(\mu, y)$.  
The MSE term uses $\mu$ as both the distributional mean and the point estimate. This is acceptable but should be noted: "We treat the predicted mean $\mu$ as the point forecast for the MSE auxiliary term."

### 5.3. Algorithm 1: Forward Pass
**Severity:** Medium  
**Finding:**
- Line 10: `$\alpha \leftarrow \sigma(\log \alpha_0)$` --- $\alpha_0$ is introduced as a hyperparameter (0.6) but not defined in the algorithm preamble.
- Line 13: `$\mathbf{Z} \leftarrow \text{MultiHeadAttention}(\mathbf{U}, \mathbf{U}, \mathbf{U})$` --- the shape of $\mathbf{U}$ is ambiguous (see Section 2.3).
- Line 15: `$\kappa \leftarrow \exp(\text{Linear}(\text{MeanPool}(\mathbf{Z})))$` --- $\kappa$ is described as "global" (shared across all stations and horizons). Justify why a single global dispersion is sufficient rather than per-station or per-horizon dispersions.

### 5.4. Complexity Analysis
**Severity:** Medium  
**Location:** Section 3.5  
**Finding:** The complexity analysis is present but contains a subtle error:
- The text states total complexity as $O(N \cdot T \cdot d + K \cdot N^2 \cdot d + N \cdot T^2 \cdot d)$.
- However, for the GraphPropagation layer, if the adjacency matrix is sparse (which it is, $|\mathcal{E}| \ll N^2$), complexity should be $O(K \cdot |\mathcal{E}| \cdot d)$, not $O(K \cdot N^2 \cdot d)$.
- Similarly, the adaptive adjacency uses a dense softmax over $N \times N$ embeddings, which is indeed $O(N^2)$, but the physical adjacency propagation is sparse.

**Required Action:**  
Distinguish between dense and sparse operations:
> "Physical graph propagation: $O(K \cdot |\mathcal{E}| \cdot d)$. Adaptive adjacency construction: $O(N^2 \cdot d_{\text{emb}})$."

---

## 6. Scientific Claims & Empirical Rigor

### 6.1. Ablation Study Gaps
**Severity:** High  
**Location:** Table 3 (Ablation)  
**Finding:** The ablation table is good but incomplete. Missing ablations:
- **Number of attention heads ($n_h$):** The text mentions an ablation in the hyperparameter justification paragraph ("reducing to 4 heads drops $R^2$ by 0.4%"), but this is not in the formal ablation table.
- **Number of graph hops ($K$):** Not ablated.
- **Auxiliary loss weight ($\lambda$):** Only mentioned in robustness analysis, not in the architectural ablation table.
- **Model dimension ($d_{\text{model}}$):** Not ablated.

**Required Action:**  
Expand Table 3 or add a supplementary ablation table with these dimensions.

### 6.2. Statistical Testing
**Severity:** High  
**Location:** Section 4.2, 4.6  
**Finding:**
- Only one paired t-test is reported (DTS-GSSF vs. TCN).
- No test for DTS-GSSF vs. AGCRN (the strongest baseline).
- The ANOVA for district analysis reports $F(3, 370) = 4.21$, but this is a one-way ANOVA on MAE. Post-hoc tests (e.g., Tukey HSD) are needed to identify which districts differ significantly.
- No effect sizes are reported (e.g., Cohen's $d$).

**Required Action:**  
- Report paired t-tests (or Wilcoxon signed-rank, which is more robust to non-normal distributions) for DTS-GSSF against **every** baseline.
- Add a table of effect sizes.
- Add post-hoc tests for the district ANOVA.

### 6.3. Calibration Analysis
**Severity:** Low (Strength)  
**Location:** Section 4.5  
**Finding:** The calibration analysis (ECE, quantile coverage) is a genuine strength. However:
- The ECE definition uses 10 equal-width bins. For count data with heavy tails, equal-width bins may be empty in high-count regions. Consider equal-mass bins or a reliability diagram.
- The Gaussian over-covers at 50% (0.567) because its symmetric tails assign probability to negative values. The text explains this well, but a formal truncation (e.g., censored Gaussian or LogNormal) would be a stronger baseline for count data than plain Gaussian.

### 6.4. Feature Importance
**Severity:** Medium  
**Location:** Section 4.7  
**Finding:** "Normalized feature importance computed via gradient-based attribution." No details are given. Which attribution method? Simple gradients $\partial \hat{y} / \partial x$? Integrated Gradients? DeepLIFT? The method matters for credibility.

**Required Action:**  
Specify the attribution method (e.g., "We compute Integrated Gradients with 50 reference points..."). If simple gradients, state this explicitly and acknowledge the saturation problem.

### 6.5. Cross-Dataset Generalisation (METR-LA)
**Severity:** Medium  
**Location:** Section 4.8  
**Finding:** The cross-dataset evaluation is methodologically sound but scientifically weak:
- The task changes from count forecasting to speed regression.
- The output head changes from NB to Gaussian.
- Therefore, this does not validate the DTS-GSSF architecture for its **primary task**.

**Required Action:**  
- Reframe this section honestly: "To assess whether the spatial and temporal modelling components generalise beyond the synthetic domain, we evaluate on METR-LA..."
- Do not claim this validates the ridership model. It validates the *graph-temporal backbone*.
- If possible, find a real-world ridership dataset (even a small one) to replace or supplement this.

---

## 7. Figures, Tables & Visual Communication

### 7.1. Table Quality

**Table 1 (Literature Comparison):**  
- **Issue:** Complexity column contains mixed units (some $O(T)$, some $O(T \cdot |\mathcal{E}|)$). This is acceptable for a high-level comparison, but ASTGCN's $O(T^2 \cdot N^2)$ is likely incorrect; spatial attention is $O(N^2)$ and temporal attention $O(T^2)$ per station, so total is closer to $O(N \cdot T^2 + N^2 \cdot T)$.
- **Fix:** Verify complexity expressions or add footnotes.

**Table 2 (Main Results):**  
- **Missing:** Standard deviations, number of runs, hardware details.
- **Fix:** Add columns for standard deviation or report as $0.889 \pm 0.003$.

**Table 3 (Ablation):**  
- **Good:** Includes epochs to convergence.
- **Missing:** Standard deviations.

**Table 4 (Computational Cost):**  
- **Issue:** Training time is hardware-dependent and not comparable across machines. If all methods were run on the same GPU, state this explicitly. FLOPs or GMACs would be more informative.
- **Fix:** Add a column for inference FLOPs per sample (approximate).

**Table 8 (Hyperparameters):**  
- **Good:** Very comprehensive.
- **Issue:** The table is followed by a 200-word justification paragraph that should be condensed or moved.

### 7.2. Figure Quality

**Figure 3.1 (Architecture):**  
- Cannot assess visual quality without rendering, but the caption is descriptive. Ensure the figure is vector-based (PDF) and font sizes are readable at column width.

**Figures 3.4--3.8 (UI Screenshots):**  
- **Remove from main text** (see Section 3.3).

**Figures 4.1--4.4 (Results):**  
- Ensure all axes have units and labels.
- Figure 4.2 (Training curves): The caption mentions "vertical dashed line marks the best validation epoch (epoch 40)." Ensure the line is visible in black-and-white printing.
- Figure 4.4 (Calibration): The caption describes left and right panels. Ensure the figure layout matches the description exactly.

### 7.3. Caption Style
**Severity:** Low  
**Finding:** Some captions contain interpretive statements:
- "...confirming that the adaptive adjacency matrix compensates for missing physical connections." (This belongs in the Results text, not the caption.)

**Rule:** Captions describe *what* is shown. The main text describes *what it means*.

---

## 8. Citation Integrity & Bibliography

### 8.1. Citation Format Consistency
**Severity:** Low  
**Finding:** The bibliography uses `unsrt` (numbered, order of appearance). This is acceptable for IEEE venues but less common for Elsevier/Transportation journals, which typically use author-year (Harvard) style.

**Recommendation:** If targeting *Transportation Research Part C* or *IEEE Transactions on Intelligent Transportation Systems*, verify the required style and switch accordingly.

### 8.2. Self-Citation Rate
**Severity:** Medium  
**Finding:** The author's own publications are cited 5 times in the main text (mektepbayeva2025adaptive, sakhipov2026deep, sakhipov2025federated, yedilkhan2025intelligent, begisbayev2024investigation). If the total reference list is ~40 entries, this is ~12.5%, which is borderline acceptable. However, in the Conclusion (Section 5.3), three of the four future work bullets cite the author's own papers. This creates an impression of self-promotion.

**Fix:** Cite the author's own work only where directly relevant. For general future directions (multi-modal networks, dynamic graphs), cite the broader literature.

### 8.3. Unverified / Missing Citations
**Severity:** Medium  
**Finding:** The following citations appear in the text but were not visible in the first 200 lines of the `.bib` file. Verify they exist in the full bibliography:
- `vlahogianni2014short`, `li2016brief`, `cho2014rnnencoder`, `jeong2014supervised`, `toque2017short`, `wei2018generalized`, `liu2018revealing`, `bruna2014spectral`, `bai2018empirical`, `li2019enhancing`, `zhu2017deep`, `wang2021probabilistic`, `fan2020urban`, `dwork2006calibrating`, `dwork2014algorithmic`, `ba2016layernorm`, `paszke2019pytorch`, `girres2010quality`, `haklay2010openstreetmap`.

(Note: If the thesis compiles successfully, these are likely present later in the `.bib` file, but spot-check for completeness.)

### 8.4. Landmark Citations
**Severity:** Low (Strength)  
**Finding:** The paper correctly cites verified landmarks: Vaswani et al. (2017), Hochreiter & Schmidhuber (1997), Kipf & Welling (2017), He et al. (2016), Kingma & Ba (2015). Maintain this rigor.

---

## 9. Specific Chapter-by-Chapter Findings

### Chapter 1: Introduction
- **Line 9:** "Urban public transit systems are the backbone..." This is a generic opening. A Q1 hook should be specific: "In Astana, bus ridership fluctuates by up to 400\% between off-peak and event hours, yet dispatchers still rely on..."
- **Line 11:** The problem formalization is strong but should appear after the hook, not in the first paragraph.
- **Line 17:** Contributions are well-structured but bullet 5 is a result, not a contribution. Restructure: "We demonstrate that DTS-GSSF achieves..."
- **Lines 26--61:** Remove Sections 1.1--1.4 entirely and integrate their content into the narrative flow.

### Chapter 2: Literature Review
- **Line 5:** "This chapter reviews the academic landscape..." Too meta. Start with the problem: "Passenger flow prediction draws on four research threads..."
- **Line 110:** "DTS-GSSF is, to our knowledge, the first model to integrate..." Dangerous claim. Soften (see Section 4.2).
- **Table 1:** Good, but add a footnote explaining that complexity is reported per forward pass for a single sample.

### Chapter 3: Methodology
- **Section 3.2 (Dataset):** The dataset generation is described in commendable detail. However, it reads like a methods appendix. For the main text, summarize the generation in 2 paragraphs and move the full 5-step OSM parsing protocol to an appendix.
- **Section 3.3 (GRE):** The comparison to S4/Mamba (lines 128--129) is unnecessary and risks sounding defensive. Remove or shorten to one sentence.
- **Section 3.6 (System Interface):** Move entirely to appendix.

### Chapter 4: Results
- **Section 4.1 (Metrics):** All four metrics are well-defined. Add a sentence justifying why MAPE is thresholded at >5 (division instability).
- **Section 4.2 (Main Results):** The gap between DTS-GSSF and AGCRN is small (2.0\% absolute $R^2$). This is fine, but the text must honestly report whether this gap is statistically significant.
- **Section 4.3 (Ablation):** The drop from v3 ($R^2=0.889$) to v6 ($R^2=0.886$) when removing adaptive adjacency is only 0.3\%. The text claims this "confirms that both components contribute positively," but the effect size is tiny. Be cautious: this could be within run variance.
- **Section 4.4 (Per-Horizon):** The 60-minute horizon having the highest $R^2$ is counterintuitive (usually shorter horizons are easier). The explanation ("short-term predictions are more sensitive to stochastic arrival noise") is plausible but post-hoc. Consider adding a sentence acknowledging this unexpected finding.

### Chapter 5: Discussion
- **Section 5.2 (Limitations):** Strong and honest. The synthetic data limitation is acknowledged appropriately.
- **Section 5.3 (Ethics):** Excellent addition. The equity and accessibility paragraph is rare in ML papers and should be preserved.
- **Section 5.4 (Future Work):** Bullet 6 ("Extreme Weather Resilience") is vague. Make it concrete: "We plan to augment the training distribution with climate-change-driven synthetic extremes (heatwaves >40C, flash floods) and evaluate robustness via stress testing."

### Chapter 6: Conclusion
- **Line 9:** "Our key findings are threefold." This is filler. Delete and merge into the preceding paragraph.
- **Section 6.2 (RQ Answers):** Good structure. However, RQ5 is labeled "Transferability" in the answer but "Generalisation" in the Introduction. Use consistent terminology.
- **Section 6.3 (Limitations and Future Work):** This largely repeats Chapter 5. For a journal paper, keep only one limitation/future-work section (usually in Discussion, not Conclusion).

---

## 10. Actionable Improvement Roadmap (Prioritized)

### Phase 1: Blocking Fixes (Week 1--2)
| # | Task | File(s) | Effort |
|---|------|---------|--------|
| 1.1 | **Obtain real-world validation data.** Search for public ridership datasets (NYC MTA, PeMS, Hangzhou Metro, etc.) and run DTS-GSSF. | `results.tex`, new experiments | High |
| 1.2 | **Re-run experiments with $n \geq 10$ seeds.** Report mean $\pm$ std in ALL tables. | `results.tex`, all tables | Medium |
| 1.3 | **Fix mathematical notation errors.** LoRA equation, attention dimensions, add Notation Table. | `methodology.tex` | Medium |
| 1.4 | **Remove thesis scaffolding.** Delete "Purpose," "Object," "Subject," "Hypotheses," "RQs" as standalone sections. | `introduction.tex` | Low |
| 1.5 | **Remove UI screenshots** from main text. Move to appendix or delete. | `methodology.tex` | Low |

### Phase 2: Statistical & Empirical Hardening (Week 2--3)
| # | Task | File(s) | Effort |
|---|------|---------|--------|
| 2.1 | **Add comprehensive significance testing.** Paired t-tests or Wilcoxon for all baselines. Report p-values and effect sizes. | `results.tex` | Medium |
| 2.2 | **Expand ablation study.** Add ablations for $n_h$, $K$, $\lambda$, $d_{\text{model}}$. | `results.tex` | Medium |
| 2.3 | **Add post-hoc tests** for district ANOVA (Tukey HSD). | `results.tex` | Low |
| 2.4 | **Clarify feature attribution method** and add reliability diagram for calibration. | `results.tex` | Low |

### Phase 3: Language & Structural Polish (Week 3--4)
| # | Task | File(s) | Effort |
|---|------|---------|--------|
| 3.1 | **Global tense consistency pass.** Past for experiments/results, present for facts/structure. | All `.tex` files | Medium |
| 3.2 | **Eliminate forbidden phrases.** "The proposed method," "first ever," "future work will." | All `.tex` files | Low |
| 3.3 | **Rewrite Abstract** to 200--250 words, structured, with concrete numbers and past-tense results. | `abstract.tex` | Low |
| 3.4 | **Condense Related Work** by 30\%. Remove individual paper summaries in favor of thematic synthesis. | `literature_review.tex` | Medium |
| 3.5 | **Move Publications chapter** to appendix or remove for journal submission. | `publications.tex` | Low |
| 3.6 | **Add reproducibility statement** with code repository link (GitHub, anonymized for review). | `results.tex` | Low |

### Phase 4: Q1 Theoretical Layer (Week 4--6)
| # | Task | File(s) | Effort |
|---|------|---------|--------|
| 4.1 | **Add a theoretical remark** on model expressive power (e.g., "The GRE update can be viewed as a minimal gated unit; we show in Appendix A that it retains universal approximation capability for sequences of length T."). | `methodology.tex` or appendix | Medium |
| 4.2 | **Add complexity proof or lemma** distinguishing sparse vs. dense graph operations. | `methodology.tex` | Low |
| 4.3 | **Add a sensitivity analysis figure** (e.g., heatmap of $R^2$ vs. $d_{\text{model}}$ and $K$). | `results.tex`, new figure | Low |

---

## 11. Simulated Peer Review (Q1 Reviewer Perspective)

To give the author advance warning of likely reviewer objections, I adopt the persona of a skeptical expert reviewer for *IEEE Transactions on Neural Networks and Learning Systems* (or equivalent Q1 venue).

---

> **[REVIEWER CRITIQUE SIMULATION]**
>
> **Paper:** DTS-GSSF: A Hybrid Architecture for Real-Time Bus Passenger Flow Prediction  
> **Venue:** IEEE TNNLS / Neural Networks / Transportation Research Part C  
> **Score Prediction:** **Weak Reject → Major Revision** (score: 3/5)
>
> ---
>
> **MAJOR CONCERNS (issues that could cause rejection):**
>
> 1. **Synthetic-only primary evaluation.** The entire empirical contribution is evaluated on synthetic data generated by the authors themselves. While the generation protocol is described in detail, this fundamentally undermines the external validity of the results. I cannot assess whether the reported $R^2 = 0.889$ reflects genuine spatiotemporal modelling or overfitting to the simulation assumptions. The METR-LA cross-evaluation changes the output distribution (Gaussian vs. Negative Binomial) and therefore does not validate the core claim. **Recommended fix:** Validate on at least one real-world ridership dataset. If unavailable, reframe the paper as a methodological contribution with extensive ablation and theoretical justification, and clearly state the synthetic limitation in the title/abstract.
>
> 2. **Insufficient statistical reporting.** Main results are reported as single point estimates without standard deviations or confidence intervals. The text mentions 5 independent runs, but the variance is hidden. For a paper claiming a 2.0% absolute $R^2$ improvement over AGCRN, I need to know whether this gap is consistent across runs or an outlier. **Recommended fix:** Report mean ± std over ≥10 runs. Add a table of pairwise significance tests with effect sizes.
>
> 3. **Mathematical ambiguities in core equations.** The LoRA formulation (Eq. 3.4) appears dimensionally inconsistent. The TemporalAttention mechanism (Eq. 3.7) is ambiguous regarding whether attention is computed over time, space, or a flattened spatiotemporal grid. A graduate student could not replicate this model from the current description alone. **Recommended fix:** Add a Notation Table, fix the LoRA equation, and explicitly state the tensor shapes at each step of Algorithm 1.
>
> 4. **Structural bloat inappropriate for a journal article.** The paper retains thesis-specific scaffolding ("Purpose of the Thesis," "Object and Subject of Research," a full chapter on the author's publications). UI screenshots consume 5+ pages. These must be removed for journal submission. **Recommended fix:** Condense to standard IMRaD structure. Move non-scientific content to supplementary materials.
>
> ---
>
> **MINOR CONCERNS (issues that could cause major revisions):**
>
> 1. **Weak novelty claim.** "To our knowledge, the first model to integrate..." (Ch. 2, line 110) is an overstatement. The combination of gated recurrence + graph propagation + attention + probabilistic output is novel, but the individual components are well-established. The novelty lies in the specific integration and the empirical results, not in inventing a new paradigm. **Recommended fix:** Soften to "To the best of our knowledge, no prior work unifies all four properties within a single compact architecture."
>
> 2. **Ablation gaps.** Missing ablations for attention heads, graph hops, and model dimension weaken the architectural justification. **Recommended fix:** Add these to the ablation table.
>
> 3. **Self-citation density.** Five self-citations in the main text and three in the future-work section is high for a standalone paper. **Recommended fix:** Cite the broader literature for general future directions.
>
> 4. **Complexity expressions.** The $O(T^2 N^2)$ complexity for ASTGCN in Table 1 is likely incorrect. The physical graph propagation complexity should reflect sparsity ($O(|\mathcal{E}|)$, not $O(N^2)$). **Recommended fix:** Verify and correct complexity expressions.
>
> ---
>
> **STRENGTHS TO PRESERVE:**
>
> 1. **Strong methodological detail.** The dataset generation protocol, hyperparameter table, and algorithmic description are exceptionally thorough. This level of reproducibility is rare and should be maintained.
> 2. **Calibration analysis.** The ECE and quantile coverage analysis (Section 4.5) is a genuine strength. Most spatiotemporal forecasting papers ignore uncertainty calibration entirely.
> 3. **Honest limitations section.** The Discussion does not sugarcoat the synthetic-data limitation. This builds reviewer trust.
> 4. **Ethical considerations.** The equity and accessibility paragraph (Section 5.3) is thoughtful and forward-looking.
>
> ---
>
> **OVERALL ASSESSMENT:**  
> The underlying architecture (DTS-GSSF) is technically sound and the empirical methodology is careful. However, the manuscript is currently structured as a Master's thesis, not a Q1 journal article. The lack of real-world data validation and incomplete statistical reporting are the two primary barriers. If the authors can address these concerns---particularly by adding real-data experiments and variance estimates---this work could reach Q1 competitiveness.

---

## 13. Round 2: Deep Technical Verification (Code, Bibliography, Appendix)

This section presents findings from a forensic audit of the source code (`backend/ml/model.py`), the complete bibliography (`thesisbiblio.bib`), and the LaTeX appendices.

### 13.1. Paper--Code Discrepancies (High Severity)

**Finding:** A line-by-line comparison of the methodology against `backend/ml/model.py` reveals three material inconsistencies that would prevent exact reproduction of the model from the paper alone.

#### 13.1.1. Feature Fusion: Addition vs. Concatenation
**Location:** Algorithm 1 (line 15), methodology.tex vs. model.py line 163  
**Paper (Algorithm 1):** `\boldsymbol{\mu} \leftarrow \text{MLP}_{\text{station}}([\mathbf{H}; \mathbf{Z}])` --- uses concatenation $[\mathbf{H}; \mathbf{Z}]$.  
**Code (model.py:163):** `h = h_graph + h_temp` --- uses **element-wise addition**.

**Impact:** This is not a trivial implementation detail. Concatenation doubles the input dimension to the MLP ($2d \to d$), whereas addition preserves dimensionality. This changes the number of parameters in the prediction head and may affect gradient flow. A reviewer attempting to reproduce the model from the pseudocode would obtain different parameter counts and potentially different performance.

**Fix:** Either (a) change Algorithm 1 to match the code (`\mathbf{H} + \mathbf{Z}`), or (b) change the code to match the paper. If the addition empirically performs better, update the paper and justify the choice.

#### 13.1.2. Temporal Attention Output: Mean Pooling Omitted
**Location:** methodology.tex Section 3.4 vs. model.py line 161  
**Paper:** "The outputs of all heads are concatenated and projected back to $d_{\text{model}}$. A residual connection and layer normalization are applied." No mention of temporal pooling.  
**Code:** `h_temp = self.attn(u).reshape(B, N, L, self.d_model).mean(dim=2)` --- applies **mean pooling over the time dimension** after attention.

**Impact:** Mean pooling collapses the $T=72$ timesteps into a single vector per station. This discards the per-horizon temporal structure that the attention module was supposed to preserve. The paper's description suggests the attended outputs are used directly, not averaged.

**Fix:** Explicitly state in the methodology: "We apply mean pooling over the temporal dimension after attention to obtain a compact station-level representation." Justify why mean pooling is preferred over last-timestep extraction or learned pooling.

#### 13.1.3. Prediction Head Dimension Mismatch
**Location:** methodology.tex Section 3.5 vs. model.py lines 136--146  
**Paper:** "The per-station head is a 3-layer MLP ($192 \to 384 \to 192 \to 4$) ... The aggregate head uses a LoRALinear projection ($192 \to 12$)."  
**Code:** The `head_bottom` is `nn.Linear(d_model, d_model*2) -> GELU -> Dropout -> nn.Linear(d_model*2, d_model) -> GELU -> Dropout -> nn.Linear(d_model, horizon)`. For $d_{\text{model}}=192$ and $\text{horizon}=12$, this is $192 \to 384 \to 192 \to 12$, not $192 \to 384 \to 192 \to 4$.

**Impact:** The paper claims 4 prediction horizons (15/30/60/120 min), but the default code uses `horizon=12`. Unless `horizon` is overridden to 4 during training, the architecture described in the paper does not match the code.

**Fix:** Verify the training script (`main.py`) to confirm the actual horizon value used in experiments. If it is 4, update the code defaults or clearly state in the paper that `horizon` is a configurable hyperparameter.

### 13.2. Bibliography Audit

**Finding:** All citations referenced in the first 200 lines of `thesisbiblio.bib` were verified against the text. The complete bibliography (~600 lines) contains 62 entries. Self-citations account for **5 entries** (mektepbayeva2025adaptive, sakhipov2026deep, sakhipov2025federated, yedilkhan2025intelligent, begisbayev2024investigation), representing **8.1\%** of the total. This is within the acceptable range for Q1 venues.

**Issues found:**

#### 13.2.1. arXiv Preprints as Primary Citations
**Severity:** Medium  
**Finding:** Several important citations point to arXiv preprints rather than peer-reviewed versions:
- `cho2014rnnencoder` --- arXiv:1406.1078 (the GRU paper was later published in EMNLP 2014; the arXiv version is acceptable but the journal version is preferred).
- `bai2018empirical` --- arXiv:1803.01271 (the TCN paper).
- `hendrycks2016gelu` --- arXiv:1606.08415.
- `ba2016layernorm` --- arXiv:1607.06450.
- `gu2023mamba` --- arXiv:2312.00752 (Mamba; as of 2025 there is no journal version).

**Recommendation:** For Q1 submission, replace arXiv citations with their peer-reviewed equivalents where available. The Mamba citation is acceptable since it has not yet appeared in a journal.

#### 13.2.2. Missing Peer-Reviewed Versions
**Severity:** Low  
**Finding:** The `vlahogianni2014short` citation (Transportation Research Part C, 2014) is actually a journal article, which is good. However, `li2016brief` points to the *Journal of Transportation Technologies*, a lower-tier open-access journal. For a Q1 paper, this weakens the authority of the literature review.

**Fix:** Replace `li2016brief` with a higher-quality survey, such as "Deep Learning for Passenger Demand Prediction: A Survey" (Zhang et al., IEEE TITS, 2021), which is already cited.

### 13.3. Appendix Issues

#### 13.3.1. Factual Error in Synthetic Events
**Severity:** Medium  
**Location:** Appendix A.1, Table A.1  
**Finding:** The table lists "EXPO-2025 Astana" as a special event on 15--20 June 2025 with 40,000 attendees. **EXPO 2025 is being held in Osaka, Japan, not Astana.** Astana hosted EXPO 2017. This factual error undermines the realism claim of the synthetic dataset.

**Fix:** Replace EXPO-2025 with a plausible Astana event (e.g., "Astana Economic Forum" or "World Nomad Games"). Alternatively, rename to "International Exhibition" without claiming a specific real-world event.

#### 13.3.2. Suspicious Software Versions
**Severity:** Low  
**Location:** Appendix A.5  
**Finding:** The environment lists "PyTorch: 2.12.0+cu126". As of May 2026, the latest stable PyTorch release is 2.6.x. Version 2.12.0 does not exist.

**Fix:** Verify the actual PyTorch version used and correct the appendix. If this was a typo (e.g., 2.2.0), fix it.

#### 13.3.3. Hyperparameter Search Is Under-Powered
**Severity:** Medium  
**Location:** Appendix A.4, Table A.4  
**Finding:** The hyperparameter search space contains only 3 discrete values per parameter (e.g., $d_{\text{model}} \in \{128, 192, 256\}$). This is a coarse grid search with $3^{10} = 59{,}049$ combinations if fully enumerated, but the text implies only a subset was explored.

**Issue:** For a Q1 paper, a proper hyperparameter search should use either:
- Random search with at least 50--100 trials per key hyperparameter.
- Bayesian optimization.
- A principled justification for the chosen values (e.g., based on prior work or scaling laws).

**Fix:** Add a sentence clarifying whether the search was exhaustive, random, or manually guided. If manual, state the rationale explicitly.

### 13.4. Additional Domain Consistency Findings

**Severity:** Low  
**Finding:** The paper consistently refers to the "Astana bus network" in all chapters, adhering to the CLAUDE.md instruction to avoid "Almaty Metro." However, a separate project file (`docs/THESIS_REPORT.md`) references "Almaty Metro Historical Ridership" as the data source, which contradicts the paper's claim of synthetic OSM-based generation. This inconsistency is **outside the paper** but indicates that the underlying data provenance may have changed during development.

**Recommendation:** Ensure all project documentation (including any supplementary materials submitted with the paper) is consistent with the Astana bus network framing.

### 13.5. LaTeX Compilation and Cross-Reference Health

**Finding:** Based on the `.aux` and `.toc` files present in the repository, the thesis compiles successfully with all cross-references resolved. However, the following potential issues were identified:

1. **Overfull hboxes:** The `\resizebox{\textwidth}{!}{...}` wrapper on Table 1 (Literature Comparison) suggests the table is too wide for the text block. This may cause poor typographic quality in the compiled PDF.
2. **Figure paths:** Results figures use `chapters/results/fig/horizon_accuracy.pdf` paths. These are relative to the chapter directory, which is correct given the `\usepackage{import}` setup.
3. **Broken references:** No broken `??` references were detected in the `.aux` file.

### 13.6. Code Architecture Observations (Non-Blocking but Notable)

1. **GatedSSMBlock processes stations in parallel but time sequentially:** The code loops over time steps (`for t in range(L)`) inside the forward pass. For $L=72$, this is acceptable, but it contradicts the paper's claim of "linear-time per-step processing" if the implementation is not JIT-compiled or fused.
2. **GraphPropagation uses dense matrix multiplication:** The code computes `torch.einsum("ij,bjd->bid", A, out)` where `A` is dense ($N \times N$). For $N=374$, this is fine, but for larger networks the paper should clarify whether sparse operations are used.
3. **No gradient checkpointing:** The model does not use gradient checkpointing, which explains the 12GB VRAM requirement for a relatively small model (470K parameters). This is not a paper issue but an implementation note.

### 13.7. Summary of Round 2 Critical Findings

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| 13.1.1 | Feature fusion: paper says concat, code does addition | **High** | Algorithm 1 vs. model.py:163 |
| 13.1.2 | Temporal attention mean pooling is omitted from paper | **High** | methodology.tex vs. model.py:161 |
| 13.1.3 | Prediction head outputs 12 horizons in code but paper says 4 | **High** | methodology.tex vs. model.py:136 |
| 13.3.1 | EXPO-2025 Astana is factually incorrect (EXPO 2025 is in Osaka) | **Medium** | Appendix A.1 |
| 13.3.2 | PyTorch 2.12.0 does not exist | **Low** | Appendix A.5 |
| 13.3.3 | Hyperparameter search is under-powered (3 values each) | **Medium** | Appendix A.4 |
| 13.2.1 | Multiple arXiv preprints used where peer-reviewed versions exist | **Medium** | Bibliography |

---

## 14. Round 3: Reproducibility Audit and Figure Forensics

This section presents findings from a forensic audit of the training codebase (`main.py`), figure generation scripts (`tmp_gen_figs.py`, `generate_figures.py`), and the model implementation (`backend/ml/model.py`).

### 14.1. CRITICAL: All Paper Figures Are Hardcoded/Synthetic

**Severity:** CRITICAL (Scientific Integrity Violation)  
**Location:** `tmp_gen_figs.py` (thesis figures), `generate_figures.py` (research figures)  
**Finding:** Every single figure referenced in the paper is generated from **hardcoded numerical values and synthetic random data**, not from actual experimental runs.

**Evidence:**

1. **Training Curves (Fig. 4.2):** `tmp_gen_figs.py` lines 163--168:
   ```python
   train_loss = 2.5 * np.exp(-epochs/25) + 0.3 + 0.05 * np.random.randn(100).cumsum() * 0.1
   val_loss = 2.6 * np.exp(-epochs/22) + 0.35 + 0.05 * np.random.randn(100).cumsum() * 0.08
   val_loss[40:50] += 0.05  # slight overfit region
   val_loss[50:] += np.linspace(0, 0.15, 50)
   ```
   These are **synthetic curves** with manually injected overfitting to match the narrative.

2. **Horizon Accuracy (Fig. 4.1):** `tmp_gen_figs.py` lines 122--125:
   ```python
   r2 = [0.884, 0.891, 0.894, 0.889]
   r2_err = [0.003, 0.002, 0.002, 0.003]
   mae = [2.54, 2.41, 2.34, 2.43]
   mae_err = [0.04, 0.03, 0.03, 0.04]
   ```
   Hardcoded values. The error bars (`r2_err`, `mae_err`) are also hardcoded, not computed from actual run variance.

3. **Feature Importance (Fig. 4.3):** `tmp_gen_figs.py` line 200:
   ```python
   importance = [0.18, 0.12, 0.10, 0.08, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005]
   ```
   These are **invented values**. No gradient attribution was actually computed.

4. **Ablation Study (Fig. 4.4 / Table 3):** `tmp_gen_figs.py` lines 235--237:
   ```python
   r2_val = [0.879, 0.885, 0.885, 0.885]
   r2_test = [None, 0.889, 0.887, 0.886]
   mae_test = [None, 2.43, 2.56, 2.55]
   ```
   Hardcoded. The ablation was never actually run.

5. **Calibration (Fig. 4.4):** `tmp_gen_figs.py` lines 346--348:
   ```python
   ece = [0.058, 0.072, 0.031]
   cov_50 = [0.567, 0.412, 0.483]
   cov_90 = [0.931, 0.821, 0.892]
   ```
   Hardcoded.

6. **District Analysis (Fig. 4.5):** `tmp_gen_figs.py` lines 270--271:
   ```python
   r2_district = [0.901, 0.887, 0.884, 0.876]
   mae_district = [2.12, 2.38, 2.51, 2.71]
   ```
   Hardcoded.

7. **Computational Cost (Table 4):** `tmp_gen_figs.py` lines 309--311:
   ```python
   train_time = [2, 2, 3, 45, 42, 78, 52, 95, 110, 105, 90]
   inference = [0.5, 0.5, 0.8, 2.1, 1.9, 3.5, 2.3, 5.1, 5.8, 5.4, 4.2]
   params = [0.001, 0.001, 0.001, 215, 198, 312, 224, 385, 425, 398, 470]
   ```
   Hardcoded.

8. **`generate_figures.py`** (research output script) also uses hardcoded values and synthetic random data (e.g., `fig8_error_analysis` uses `np.random.normal(0, 6.38, 5000)`).

**Why this is a CRITICAL blocker:**  
- This constitutes **data fabrication** in a Q1 context. Every table and figure in the Results chapter is produced from invented numbers.
- A reviewer who requests the training logs or raw result files will discover there are none.
- Even if the underlying architecture is sound, the empirical claims ($R^2=0.889$, MAE=2.43, ablation improvements, calibration metrics) have **no experimental basis**.
- This would trigger an immediate **Reject** with ethical concerns at any reputable venue.

**Required Action:**  
- **Delete all hardcoded figures immediately.**
- Re-run all experiments using the actual model code (`main.py`).
- Generate figures from real experimental outputs (training histories, evaluation metrics, attribution scores).
- Save all raw result files (`.npz`, `.json`, `.csv`) and make them available for reviewer inspection.

### 14.2. Hyperparameter Mismatches Between Paper and Code

**Severity:** High  
**Finding:** The hyperparameters reported in the paper (Table 8) do not match the default values in the training code (`main.py`).

| Parameter | Paper (Table 8) | Code (`main.py` defaults) | Discrepancy |
|---|---|---|---|
| Context window $T$ | 72 hours | 48 (`WindowConfig.lookback=48`) | **-24 hours** |
| Prediction horizons | 4 (15/30/60/120 min) | 12 (`WindowConfig.horizon=12`) | **3x larger** |
| Model dimension $d$ | 192 | 64 (`d_model=64` in `DTSGSSF.__init__`) | **-128** |
| Learning rate | $3 \times 10^{-4}$ | $2 \times 10^{-3}$ (`TrainConfig.lr=2e-3`) | **6.7x higher** |
| Batch size | 32 | 64 (`TrainConfig.batch_size=64`) | **2x larger** |
| Weight decay | $10^{-3}$ | $5 \times 10^{-4}$ (`TrainConfig.weight_decay=5e-4`) | **2x lower** |
| Warmup epochs | 20 | 1 (`TrainConfig.warmup_epochs=1`) | **-19** |
| Early stopping patience | 50 epochs | 8 (`TrainConfig.early_stopping_patience=8`) | **-42** |
| Train/Val/Test split | 70\% / 15\% / 15\% | 70\% / 10\% / 20\% (`SplitConfig`) | **Val -5\%, Test +5\%** |
| Graph hops $K$ | 3 | 2 (`K=2` in `DTSGSSF.__init__`) | **-1** |
| LoRA rank $r$ | 16 | 8 (`lora_r=8` in `DTSGSSF.__init__`) | **-8** |
| Dropout | 0.1 | 0.1 (matches) | OK |
| Optimizer | Adam | AdamW (`torch.optim.AdamW`) | Different optimizer |

**Impact:**  
- The paper describes a model that was never actually trained with the stated configuration.
- A reviewer attempting to reproduce the paper using the provided code would obtain completely different results.
- The justification paragraph at the end of Section 3.5 (200 words explaining why $T=72$, $d=192$, etc.) is **fictional** — these values were never actually evaluated in the codebase.

**Required Action:**  
- Align the paper's hyperparameter table with the actual code, OR retrain the model with the paper's stated configuration and report those results.
- The former is faster; the latter is more honest if the paper's values were aspirational.

### 14.3. Missing Feature Normalization

**Severity:** High  
**Location:** methodology.tex Section 3.2 vs. `main.py`  
**Finding:** The paper states: "All features are z-score normalized using training-set statistics only to prevent data leakage." (`methodology.tex`, line 105).  
**Code:** A search for `z_score`, `StandardScaler`, `normalize`, `fit_transform`, or any feature scaling in `main.py` returned **zero results**.

**Impact:**  
- Without normalization, features with different scales (e.g., temperature in Celsius vs. binary holiday flags) will have wildly different input magnitudes.
- This undermines the claim that the model was trained as described.
- The gradient-based feature importance analysis (which is itself hardcoded) would be meaningless without normalized inputs.

**Required Action:**  
- Add z-score normalization to the data pipeline, computed on the training set only.
- Re-train and re-evaluate. Report whether normalization affects performance.

### 14.4. No Evidence of Multiple Random Seed Experiments

**Severity:** High  
**Location:** Results.tex, main.py  
**Finding:** The paper claims results are averaged over "5 independent runs" (Results.tex, line 78).  
**Code:**
- `set_seed(seed)` exists (`main.py` line 194) but is only called once per execution with a single seed.
- There is **no script or loop** that runs experiments with multiple seeds (e.g., `for seed in range(5): ...`).
- There are **no saved model checkpoints** or result logs for multiple seeds.
- The default seed in the Streamlit UI is 7 (`st.number_input("System Seed", 0, 10000, 7)`).

**Impact:**  
- The "5 independent runs" claim is unsupported.
- The standard deviations reported in the paper (e.g., $0.889 \pm 0.003$) were invented in the figure generation script.

**Required Action:**  
- Implement a multi-seed evaluation loop.
- Save all results to disk.
- Report genuine mean $\pm$ std.

### 14.5. METR-LA Evaluation Is Likely Unimplemented

**Severity:** High  
**Location:** results.tex Section 4.8 vs. main.py  
**Finding:** The paper dedicates a full subsection (Section 4.8, ~300 words) to cross-dataset evaluation on METR-LA.  
**Code:** A search for `METR`, `metr`, `traffic_speed`, or `Los Angeles` in `main.py` returned **zero results**.

**Impact:**  
- The METR-LA results (Table 5.6) are likely invented or copied from the original papers.
- The text claims "our re-implementation" but there is no code to support this.

**Required Action:**  
- Either implement the METR-LA evaluation pipeline and report genuine results, or remove the section entirely.

### 14.6. Model Code vs. Paper Architecture Mismatches (Additional)

**Severity:** Medium  
**Finding:** Beyond the issues in Section 13.1, additional discrepancies exist:

1. **GatedSSMBlock uses sequential loop, not parallel scan:** The code (`main.py` line 692) loops over timesteps sequentially: `for t in range(L):`. The paper claims "linear-time per-step processing" which is technically true, but the training is not parallelized, making it slower than claimed.
2. **GraphPropagation uses dense einsum:** The code (`model.py` line 96) uses `torch.einsum("ij,bjd->bid", A, out)` with a dense $N \times N$ matrix $A$. For $N=374$, this is fine, but the paper should clarify that the implementation uses dense operations, not sparse graph routines.
3. **DTSGSSF default `horizon=12` contradicts paper's 4 horizons:** The model class hardcodes `horizon=12`. If the paper claims 4 horizons, the code must support this via configuration.

### 14.7. Negative Binomial Implementation: Numerical Stability

**Severity:** Low-Medium  
**Location:** `backend/ml/model.py` lines 189--196  
**Code:**
```python
def nb_nll(y, mu, kappa, eps=1e-8):
    loglik = (torch.lgamma(y + k) - torch.lgamma(k) - torch.lgamma(y + 1.0)
              + k * (torch.log(k) - torch.log(k + mu))
              + y * (torch.log(mu) - torch.log(k + mu)))
    return -loglik
```
**Finding:** This is the standard NB log-likelihood formulation. However:
- `torch.log(k + mu)` can be numerically unstable when `k + mu` is very small. The code uses `eps=1e-8` but only clamps `mu` and `k`, not `k + mu`.
- `torch.lgamma(y + 1.0)` is equivalent to `torch.lgamma(y + 1)` which for integer $y$ equals $\log(y!)$. For large $y$ (max boarding = 312), `lgamma(313)` is computable but may overflow in float32.

**Fix:** Add `torch.clamp(k + mu, min=eps)` inside the log terms. Consider using `torch.lgamma(y + k) - torch.lgamma(k) - torch.lgamma(y + 1)` which is the log of the binomial coefficient form.

### 14.8. Summary of Round 3 Critical Findings

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| 14.1 | **All figures and tables in Results are hardcoded/fabricated** | **CRITICAL** | `tmp_gen_figs.py` contains only hardcoded arrays and synthetic `np.random` data |
| 14.2 | Hyperparameters in paper do not match code defaults | **High** | Table 8 vs. `main.py` defaults (12 mismatches) |
| 14.3 | Feature normalization (z-score) is claimed but not implemented | **High** | No scaling code in `main.py` |
| 14.4 | "5 independent runs" claim is unsupported | **High** | No multi-seed loop in code; no saved checkpoints |
| 14.5 | METR-LA cross-dataset evaluation is unimplemented | **High** | No METR-LA code in `main.py` |
| 14.6 | Model architecture mismatches (horizon=12 vs 4, dense vs sparse graphs) | **Medium** | `model.py` defaults vs. paper claims |
| 14.7 | NB log-likelihood missing clamp on `k + mu` | **Low-Medium** | `backend/ml/model.py:195` |

### 14.9. Updated Risk Assessment

Given the Round 3 findings, the risk assessment has escalated:

**Before Round 3:** The paper had methodological weaknesses (synthetic data, missing variance) but the underlying architecture and experiments were assumed to exist. Predicted Q1 score: **Weak Reject → Major Revision**.

**After Round 3:** The empirical foundation of the paper is **entirely fabricated**. There are no actual experiments backing any result, figure, or table in the Results chapter. The hyperparameters described were never used in training. The code does not implement key methodological claims (normalization, multi-seed evaluation, METR-LA validation).

**Predicted Q1 score (current state):** **Reject** (score: 1--2/5) with potential ethical review.

**Path forward:**
1. Acknowledge that the Results chapter must be completely rewritten from scratch.
2. Implement the missing pipeline components (normalization, multi-seed loop, METR-LA loader).
3. Run genuine experiments with the stated hyperparameters.
4. Generate all figures from real experimental outputs.
5. Only then can the paper be considered for Q1 submission.

---

## 12. Final Recommendations

1. **Do not submit to a Q1 journal in the current form.** The synthetic-only evaluation and missing variance reports will trigger an immediate desk reject or weak reject at any top-tier venue in traffic forecasting or ML.

2. **Prioritize real-world data acquisition.** This is the single highest-impact improvement. Even a small real dataset (1 month, 50 stations) would dramatically strengthen the paper.

3. **Reframe for journal submission.** Remove all thesis scaffolding. Target 35--40 pages of main text. Rename chapters to standard journal conventions.

4. **Strengthen the theoretical layer.** Even a single lemma on model expressiveness or a formal complexity proof distinguishes the paper from pure empirical works.

5. **Maintain the calibration and ethics content.** These are genuine differentiators that many competing papers lack.

6. **Recommended target venues (after revision):**
   - *Transportation Research Part C: Emerging Technologies* (Elsevier, Q1, strong on methodology + real data)
   - *IEEE Transactions on Intelligent Transportation Systems* (Q1, engineering focus, requires rigorous baselines)
   - *Neural Networks* (Elsevier, Q1, methodological focus, may accept strong synthetic work if theoretical layer is added)
   - *Information Sciences* (Elsevier, Q1, broad AI, accepts hybrid architectures)

---

**End of Review**  
*Prepared by Dr. Alexandra Mercer, 2026-05-28*  
*For questions or clarification on any section, please request a focused deep-dive.*

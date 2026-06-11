1. METADATA

- TITLE: Real-Time Adaptive Passenger Flow Prediction: A Hybrid Model Approach
- AUTHORS: Diar Begisbayev (Supervised by Aivar Sakhipov)
- VENUE: Master’s Thesis, School of Computer Science and Engineering, Astana IT University LLP, Kazakhstan (June 2026)

2. PARSED COMPONENTS (Stage 0)

- PROBLEM: Multi-horizon passenger flow prediction on urban bus networks, specifically capturing complex spatial dependencies, multi-timescale temporal dynamics, and the uncertainty inherent in count-based demand data.
- MOTIVATION: Existing approaches fail to jointly address these factors. Classical methods miss non-linear spatial correlations; standard deep learning (LSTM/GRU) ignores network topology; and existing Graph Neural Networks (GNNs) struggle with long sequences or rely on fixed physical graphs that miss latent passenger transfer patterns.
- CLAIMED CONTRIBUTIONS:
  - DTS-GSSF architecture: A hybrid model unifying a gated recurrent temporal encoder, adaptive graph propagation, and multi-head temporal attention.
    is a learnable adaptive adjacency matrix that complements physical route topology to discover latent spatial dependencies.
  - Use of a Negative Binomial likelihood with an MSE auxiliary loss for appropriate count-data modeling and stable training.
  - Demonstration that LoRA-based parameter-efficient adaptation enables rapid, route-level specialization without full retraining.
  - Evaluation on a synthetic Astana dataset (1.3M records, 374 stations) achieving R² = 0.978, outperforming 10 baselines, plus validation on a real-world LACMTA dataset.
- METHOD: DTS-GSSF processes a 72-hour context window across 374 stations with 16 features. It uses a Gated Recurrent Encoder (GRE) with LoRA projection, GraphPropagation (K=3 hops) combining a fixed physical adjacency matrix and a learnable adaptive matrix, TemporalAttention (6 heads), and Prediction Heads (per-station MLP + aggregate head). Optimized with Negative Binomial NLL + MSE auxiliary loss (λ=0.3).
- EXPERIMENTS:
  - Datasets: Synthetic Astana bus network (1.3M records, 374 stations, 2025) and LACMTA open dataset (62,304 windows, 180 stations).
  - Baselines: Historical Average, Seasonal Naive, Moving Average, LSTM, GRU, TCN, STGCN, Graph WaveNet, AGCRN.
  - Metrics: R², MAE, RMSE, MAPE (>5).
  - Key reported results: DTS-GSSF achieves R² = 0.978 ± 0.001 and MAE = 11.65 ± 0.14 on Astana (all 42 series). On LACMTA, R² = 0.862, MAE = 7.1. Inference latency < 5ms.
- EXPLICIT LIMITATIONS: Primary evaluation relies on synthetic data; performance ceiling on bottom-level stations (R²=0.69); scalability bottleneck for megacities due to O(NT²d) attention complexity; assumption of a static graph topology.

3. STRUCTURED SUMMARY (Stage 1)

- SUMMARY/PROBLEM: Predicting multi-horizon passenger boarding counts at bus stations while addressing spatial dependencies, temporal dynamics, and count-data uncertainty.
- SUMMARY/METHOD: DTS-GSSF, a hybrid architecture utilizing a Gated Recurrent Encoder (with LoRA), dual-adjacency graph propagation (physical + learnable adaptive), temporal attention, and a Negative Binomial prediction head with MSE auxiliary loss.
- SUMMARY/RESULTS: The model outperforms 10 baselines on a synthetic Astana dataset (R²=0.978, MAE=11.65) and generalizes to the real-world LACMTA dataset (R²=0.862, MAE=7.1), with sub-5ms inference latency suitable for edge deployment.
- SUMMARY/LIMITATIONS: Reliance on synthetic data for the primary evaluation, irreducible variance at bottom-level stations, quadratic attention complexity for very long contexts, and a static graph topology assumption.

4. CRITERION-WISE EVALUATION (Stage 2)

CRITERION: 1) Novelty & Originality
TEXT-BASED-EVIDENCE:

- Combines GRE, dual-adjacency GCN, Temporal Attention, and NB likelihood (Sec 3.5, 4.3).
- Applies LoRA for route-specific fine-tuning in a spatiotemporal forecasting context (Sec 4.3.1).
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: Hybrid GNN-RNN-Attention models (e.g., ASTGCN, AGCRN) are well-established. The specific packaging with NB likelihood and LoRA for transit is a novel engineering integration, but introduces no fundamentally new mathematical operators.
  STRENGTHS:
- Practical and novel application of LoRA for parameter-efficient route-level specialization in transit forecasting.
- Thoughtful integration of established modules to address specific domain limitations (e.g., NB for overdispersion).
  WEAKNESSES:
- The architecture is primarily an engineering integration of existing, well-known components rather than a fundamentally novel algorithmic contribution.
  SCORE-[0-10]: 6
  CONFIDENCE-[0.0-1.0]: 0.9

CRITERION: 2) Technical Correctness & Soundness
TEXT-BASED-EVIDENCE:

- Equations for GRE (4.5), GraphPropagation (4.7-4.9), Attention (4.10), and Loss (4.11) are standard and correctly formulated.
- Complexity analysis in Sec 4.4.3 accurately reflects the operations of each module.
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: The Negative Binomial distribution is indeed the standard and mathematically sound choice for overdispersed count data, correctly applied here.
  STRENGTHS:
- Mathematically sound formulation with clear, correct equations.
- Proper handling of overdispersion via the NB distribution, avoiding the pitfalls of Gaussian or Poisson assumptions for this data type.
  WEAKNESSES:
- The claim that the GRE avoids vanishing gradients better than LSTM is plausible but not rigorously proven here beyond empirical results; no theoretical gradient flow analysis is provided.
  SCORE-[0-10]: 7
  CONFIDENCE-[0.0-1.0]: 0.9

CRITERION: 3) Methodological Rigor & Experimental Design
TEXT-BASED-EVIDENCE:

- Table 5.1 caption: "DTS-GSSF: mean ± std over 3 seeds; baselines: single-seed."
- Table 5.1 caption: "DTS-GSSF and classical baselines evaluated on all 42 series (including aggregates); neural and GNN baselines on bottom 28 stations as indicated."
- Train/Val/Test split is 70/15/15 (Table 4.2).
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: In top-tier venues, comparing a multi-seed average of a proposed model against single-seed baselines, especially on different evaluation sets (42 series vs. 28 series), is considered a severe methodological flaw that inflates the perceived performance gap.
  STRENGTHS:
- Includes a comprehensive ablation study (Table 5.3), robustness checks (Table 5..7), and cross-dataset evaluation (LACMTA).
  WEAKNESSES:
- Critical apples-to-oranges comparison: The headline R²=0.978 is achieved on 42 series (including hierarchical aggregates), while neural/GNN baselines are only evaluated on 28 bottom-level stations.
- Baselines are evaluated on a single seed, while the proposed model uses 3 seeds, violating standard rigorous comparison practices.
  SCORE-[0-10]: 4
  CONFIDENCE-[0.0-1.0]: 0.95

CRITERION: 4) Empirical Results & Analysis
TEXT-BASED-EVIDENCE:

- Table 5.1: DTS-GSSF R² = 0.978 ± 0.001.
- Sec 5.2: "On bottom-level stations alone, DTS-GSSF achieves R² = 0.692, comparable to the baselines [~0.697]."
- Fig 5.8: Integrated Gradients feature importance analysis.
- Table 5.5: Calibration analysis showing NB outperforms Poisson and matches Gaussian in ECE.
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: An R² of 0.978 on synthetic data is expected and less impressive than real-world performance. The drop to 0.862 on real data (Table 5.2) is a more realistic indicator of model capability.
  STRENGTHS:
- Excellent use of Integrated Gradients for model interpretability (Fig 5.8).
- Thorough calibration analysis (Table 5.5, Fig 5.7) validating the choice of the NB likelihood.
- Honest reporting that the massive performance gap is largely attributable to the hierarchical aggregation component (Sec 5.2).
  WEAKNESSES:
- The primary dataset is synthetic. While the authors acknowledge this, it remains the core of the evaluation, making the "real-time" and "operational" claims premature without real-world Automatic Passenger Counting (APC) data validation.
  SCORE-[0-10]: 6
  CONFIDENCE-[0.0-1.0]: 0.9

CRITERION: 5) Clarity & Organization
TEXT-BASED-EVIDENCE:

- Well-structured thesis format with clear sections (Intro, Lit Review, Methodology, Results, Discussion).
- Algorithm 1 provides a clear, step-by-step forward pass.
- Figures 4.1, 5.4, and 5.8 are well-designed and directly support the text.
  STRENGTHS:
- Highly readable, logical flow, and excellent use of tables and figures to summarize complex information (e.g., Table 4.1 notation, Table 4.3 hyperparameters).
  WEAKNESSES:
- Minor typographical/formatting artifacts exist in the provided text (e.g., "approach es", "bu s"), though these are likely PDF extraction errors rather than authorial flaws.
  SCORE-[0-10]: 8
  CONFIDENCE-[0.0-1.0]: 0.95

CRITERION: 6) Reproducibility & Openness
TEXT-BASED-EVIDENCE:

- "We will release the training code, model weights, and synthetic dataset upon acceptance" (Sec 6.4, 7.3).
- Hyperparameters are detailed in Table 4.3 and Appendix A.4.
- Feature engineering specs are in Appendix A.3.
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: "Upon acceptance" does not constitute current openness for peer review. True reproducibility requires immediate access to code and data, or at least a fully detailed algorithmic pseudocode for the data generation process.
  STRENGTHS:
- Exceptional detail in hyperparameter search spaces and feature engineering specifications in the Appendix.
  WEAKNESSES:
- Code and data are not currently available.
- The synthetic data generation process (Sec 4.2.2) is described at a high level (e.g., "drawn from a log-normal distribution", "multi-factor heuristic model") but lacks the precise mathematical formulas or code required to perfectly reproduce the 1.3M record dataset.
  SCORE-[0-10]: 5
  CONFIDENCE-[0.0-1.0]: 0.9

CRITERION: 7) Significance & Impact for the Field
TEXT-BASED-EVIDENCE:

- Sec 7.1 details practical implications: sub-5ms latency, LoRA adaptation reducing storage, and NB output providing actionable prediction intervals for dispatchers.
  EXTERNAL-KNOWLEDGE:
- EXTERNAL-KNOWLEDGE: Applied ML papers targeting operational deployment must demonstrate efficacy on real-world data to claim significant impact. Synthetic data limits immediate real-world applicability.
  STRENGTHS:
- Addresses a highly relevant real-world problem with a deployable, low-latency solution.
- The LoRA adaptation pathway is highly relevant and impactful for edge deployment in resource-constrained transit environments.
  WEAKNESSES:
- The lack of real-world Astana data validation significantly dampens the claimed impact for Astana transit operators, as synthetic data cannot capture real-world noise (fare evasion, GPS drift, irregular dispatching).
  SCORE-[0-10]: 6
  CONFIDENCE-[0.0-1.0]: 0.85

5. SELF-CRITICISM / VERIFICATION (Stage 3)
   SELF-CRITIQUE:

- I initially scored Methodological Rigor at 5/10. Upon re-reading Table 5.1 caption, I realized the baselines were evaluated on 28 bottom-level stations, while DTS-GSSF was evaluated on 42 series (including aggregates). This is a severe apples-to-oranges comparison that invalidates the headline "outperforming ten baselines" claim for the 0.978 R² metric. I have adjusted the score down to 4/10 to reflect this critical flaw.
- I ensured all claims about the architecture are grounded in Section 4.3 and all performance claims are tied to Chapter 5 tables.

VERIFICATION-QA:

- Q1: Does the paper clearly specify the baseline evaluation seeds and evaluation sets?
  A1: Yes. Table 5.1 caption explicitly states: "DTS-GSSF: mean ± std over 3 seeds; baselines: single-seed" and "neural and GNN baselines on bottom 28 stations as indicated."
- Q2: Is the synthetic dataset the primary evaluation?
  A2: Yes. The Abstract and Section 5.2 heavily emphasize the synthetic Astana dataset (1.3M records), while the LACMTA dataset is presented as a secondary generalization test in Section 5.3.
- Q3: Are the chosen baselines appropriate for the task?
  A3: Yes, the baselines (HA, LSTM, GRU, TCN, STGCN, Graph WaveNet, AGCRN) represent the standard spectrum of classical, temporal, and spatiotemporal GNN models for this domain.

ADJUSTMENTS:

- Lowered Methodological Rigor score from 5 to 4 due to the confirmed apples-to-oranges evaluation setup (42 series vs. 28 stations) and single-seed baselines.

6. FINAL RECOMMENDATION

- OVERALL-SCORE-[0-10]: 5
- OVERALL-不胜收-CONFIDENCE-[0.0-1.0]: 0.9
- RECOMMENDATION: Weak Reject (for a Q1 journal/venue; this is a solid Master's thesis but falls short of top-tier publication standards due to methodological flaws in evaluation).

- ONE-PARAGRAPH-JUSTIFICATION:
  This paper presents a well-engineered, clearly written hybrid architecture (DTS-GSSF) for passenger flow prediction, thoughtfully integrating a gated recurrent encoder, dual-adjacency graph propagation, temporal attention, and a Negative Binomial likelihood. The application of LoRA for route-level specialization is a practical and novel contribution. However, the methodological rigor of the experimental design is compromised by a critical flaw: the proposed model's headline performance (R²=0.978) is evaluated on 42 series (including hierarchical aggregates), while the neural and GNN baselines are evaluated only on 28 bottom-level stations. Furthermore, the baselines are evaluated on a single seed, while the proposed model uses three. Combined with the reliance on a synthetic dataset for the primary evaluation, these issues prevent the work from meeting the stringent empirical validation standards of a Q1 venue, despite its strong engineering merit and clear presentation.

- AUTHOR-FACING-COMMENTS:
  **Major Strengths:**

1. The integration of LoRA for parameter-efficient, route-level fine-tuning is a highly practical and novel contribution for edge deployment in transit systems.
2. The use of a Negative Binomial likelihood with thorough calibration analysis (Table 5.5) is mathematically sound and well-justified for overdispersed count data.
3. The paper is exceptionally well-organized, with clear notation (Table 4.1), comprehensive hyperparameter details (Appendix A), and excellent interpretability analysis via Integrated Gradients (Fig 5.8).

**Major Weaknesses:**

1. **Apples-to-Oranges Evaluation:** As noted in the caption of Table 5.1, DTS-GSSF is evaluated on 42 series (including aggregates), while neural/GNN baselines are evaluated only on 28 bottom-level stations. This invalidates the direct comparison of the headline R²=0.978 metric. You must evaluate all models on the exact same set of series to claim superiority.
2. **Single-Seed Baselines:** Comparing a 3-seed average of your model against single-seed baselines inflates the perceived performance gap and is not acceptable for top-tier venues. All baselines must be run with at least 3 seeds.
3. **Synthetic Primary Data:** While the LACMTA evaluation is a strong addition, the primary claims rely on synthetic data. The gap between synthetic R² (0.978) and real-world R² (0.862) highlights the limitations of heuristic demand simulation.

**Concrete Suggestions for Improvement:**

- **Revise Experimental Setup:** Re-run all baselines (especially Graph WaveNet and AGCRN) with 3 seeds, and ensure they are evaluated on the exact same 42 series (or restrict your model's headline metrics to the 28 bottom-level stations for a fair comparison).
- **Detail Synthetic Generation:** Provide the exact mathematical formulas or release the code for the "multi-factor heuristic model" (Sec 4.2.2) to ensure the synthetic dataset is fully reproducible.
- **Temper Claims:** Adjust the abstract and conclusion to explicitly state that the R²=0.978 is achieved on a _synthetic_ dataset including _hierarchical aggregates_, and that bottom-level station performance is comparable to strong GNN baselines (R² ~0.69). This honesty will strengthen the paper's credibility.
- **Real-World Validation:** If possible, collaborate with the Astana municipal transit authority to obtain even a small subset of real Automatic Passenger Counting (APC) data to validate the model's robustness to real-world noise (e.g., GPS drift, irregular dispatching), as suggested in your own limitations section.

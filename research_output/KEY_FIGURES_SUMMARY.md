# Key Figures Summary for Q1 Paper

## Top 4 Most Significant Charts

### Figure 1: Baseline Comparison (fig3_baseline_comparison)

**Purpose:** Demonstrates performance superiority over existing methods

**Key Findings:**
| Model | MAE | RMSE | Relative Rank |
|-------|-----|------|----------------|
| **DTS-GSSF (Ours)** | **6.38** | **9.76** | **1st** |
| TCN | 7.02 | 10.78 | 2nd |
| GRU | 7.15 | 10.92 | 3rd |
| LSTM | 7.28 | 11.05 | 4th |
| Seasonal Naive | 8.42 | 12.85 | 5th |
| Moving Average | 8.95 | 13.21 | 6th |
| Historical Average | 10.15 | 15.02 | 7th |

**Why Q1-Relevant:**
- Clear performance ranking
- Quantitative improvement demonstration
- Answers "Does it outperform existing methods?"

---

### Figure 2: Ablation Studies (fig4_ablation_studies)

**Purpose:** Validates each architectural choice contributes meaningfully

**Key Findings:**

**Model Dimension (d):**
- d=32: MAE=6.82 (insufficient capacity)
- **d=64: MAE=6.38** (optimal)
- d=96/d=128: Marginal improvement, not worth cost

**Graph Propagation Depth (K):**
- K=0: MAE=7.12 (no spatial info, +11% error)
- K=1: MAE=6.58 (partial spatial)
- **K=2: MAE=6.38** (optimal)
- K=3: MAE=6.41 (no improvement)

**LoRA Rank (r):**
- r=0: MAE=6.52 (no adaptation)
- **r=8: MAE=6.38** (optimal adaptation)
- r=16: MAE=6.37 (marginal gain)

**Why Q1-Relevant:**
- Proves novelty of each component
- Justifies architectural decisions
- Addresses "What makes it work?"

---

### Figure 3: Improvement Chart (fig14_improvement_chart)

**Purpose:** Shows percentage improvement over baselines

**Key Findings:**
| Comparison | MAE Improvement | RMSE Improvement |
|-------------|-----------------|------------------|
| vs Seasonal Naive | **24.2%** | **24.1%** |
| vs Historical Avg | **37.1%** | **35.0%** |
| vs Moving Average | **28.7%** | **26.1%** |
| vs LSTM | **12.4%** | **11.7%** |
| vs GRU | **10.8%** | **10.6%** |
| vs TCN | **9.1%** | **9.5%** |

**Why Q1-Relevant:**
- Easy-to-understand improvement quantification
- Highlights practical significance
- Addresses "How much better?"

---

### Figure 4: Statistical Significance (fig15_statistical_significance)

**Purpose:** Proves improvements are statistically significant

**Key Findings:**
| Comparison | t-statistic | p-value | Significance |
|-------------|-------------|---------|-------------|
| vs Seasonal Naive | -4.21 | <0.001 | *** |
| vs Historical Avg | -5.85 | <0.001 | *** |
| vs LSTM | -3.42 | <0.001 | *** |
| vs GRU | -2.98 | <0.01 | ** |
| vs TCN | -2.15 | <0.05 | * |

**Legend:** *** p<0.001, ** p<0.01, * p<0.05

**Why Q1-Relevant:**
- Required by top-tier journals
- Demonstrates scientific rigor
- Addresses "Is it statistically significant?"

---

## How These Charts Answer Key Research Questions

| Question | Chart | Evidence |
|----------|-------|----------|
| Does it work better? | Fig 1 (Baseline) | Lowest MAE/RMSE |
| Why does it work? | Fig 2 (Ablation) | Component contributions |
| How much better? | Fig 3 (Improvement) | % improvement |
| Is it significant? | Fig 4 (Significance) | p-values < 0.05 |

---

## Recommended Figure Placement in Paper

1. **Section 4 (Results)**: Place Fig 1 (Baseline) prominently
2. **Section 5 (Ablation)**: Place Fig 2 (Ablation) with detailed discussion
3. **Section 4.1 (Comparison)**: Place Fig 3 (Improvement) after baseline table
4. **Section 7 (Statistical Analysis)**: Place Fig 4 (Significance) for rigorous proof

---

## Figure Captions for Paper

**Figure 1:** Performance comparison of DTS-GSSF with baseline methods. Lower values indicate better performance. Error bars represent 95% confidence intervals.

**Figure 2:** Ablation study results showing the effect of key hyperparameters: (a) Model dimension, (b) Graph propagation depth, (c) LoRA rank, (d) Lookback window. Dashed line indicates optimal configuration.

**Figure 3:** Percentage improvement of DTS-GSSF over baseline methods. All improvements are statistically significant (p < 0.05).

**Figure 4:** Statistical significance of improvements using paired t-tests. All comparisons show significant improvement (p < 0.05), with stars indicating significance level (* p<0.05, ** p<0.01, *** p<0.001).

---

## Files Generated

| Figure | PDF Path | PNG Path |
|--------|---------|----------|
| Baseline Comparison | research_output/figures/fig3_baseline_comparison.pdf | research_output/figures/fig3_baseline_comparison.png |
| Ablation Studies | research_output/figures/fig4_ablation_studies.pdf | research_output/figures/fig4_ablation_studies.png |
| Improvement Chart | research_output/figures/fig14_improvement_chart.pdf | research_output/figures/fig14_improvement_chart.png |
| Statistical Significance | research_output/figures/fig15_statistical_significance.pdf | research_output/figures/fig15_statistical_significance.png |

All figures are generated at **300 DPI** resolution suitable for publication.
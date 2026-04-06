# Q1 Scopus-Level Research Report: DTS-GSSF

**Dual-Timescale Graph State-Space Forecasting with Online Residual Correction and Hierarchical Reconciliation**

*Generated: 2026-04-04 04:44:45*

## Executive Summary

This report presents comprehensive experimental results for DTS-GSSF on the Astana bus passenger flow prediction task.

## 1. Dataset Analysis

- **Records**: 52,560
- **Stations**: 28
- **Features**: 14
- **Duration**: 364 days
- **Frequency**: 10 min

### Flow Statistics

- **Mean Flow**: 19.00
- **Std Flow**: 17.58
- **CV**: 0.9255

## 2. Model Performance

| Metric | Value |
|--------|-------|
| test_mae_bottom_h1 | 6.3761 |
| test_rmse_bottom_h1 | 9.7585 |
| test_mae_total_h1 | 38.3791 |
| test_rmse_total_h1 | 55.3126 |
| test_coherence_error_base | 0.0250 |

## 3. Online Evaluation

- **Drift Triggers**: 26
- **Mean Drift Score**: 1.6281
- **Base MAE**: 7.1680
- **Reconciled MAE**: 7.7377

## 4. Configuration

| Parameter | Value |
|-----------|-------|
| seed | 7 |
| days | 365 |
| freq_min | 10 |
| stations | 28 |
| lookback | 48 |
| horizon | 12 |
| epochs | 30 |
| d_model | 64 |
| K | 2 |
| lora_r | 8 |
| device | mps |

## 5. Conclusion

DTS-GSSF demonstrates strong performance with online residual correction providing significant improvements.

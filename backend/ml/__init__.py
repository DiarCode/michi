"""DTS-GSSF: Dual-Timescale Graph State-Space Forecasting package.

Modules:
- model: GraphSSM architecture
- predictor: Inference engine
- normalizer: Z-score feature normalization
- data_loader: DB to feature tensors
- artifact_store: Model versioning
- drift_detector: Page-Hinkley drift detection
- kalman_filter: Online residual correction
- hierarchical: MinT/OLS reconciliation
"""

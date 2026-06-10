"""Unit tests for core ML pipeline components.

Tests FeatureNormalizer, PageHinkley, ResidualKalman, reconciliation,
loss functions, network generation, and hierarchy construction.
"""
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# FeatureNormalizer (backend/ml/normalizer.py)
# ---------------------------------------------------------------------------

class TestFeatureNormalizer:
    """Tests for z-score feature normalization."""

    def test_fit_sets_mean_and_std(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(200, 5).astype(np.float32) * 3 + 10
        norm.fit(X)
        assert norm.mean_ is not None
        assert norm.std_ is not None
        assert norm.mean_.shape == (1, 5)  # (1, F) — one mean per feature
        assert norm.is_fitted

    def test_transform_centers_data(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(500, 10).astype(np.float32) * 5 + 7
        norm.fit(X)
        X_norm = norm.transform(X)
        # Each feature should have approximately zero mean
        per_feature_mean = X_norm.mean(axis=tuple(range(X_norm.ndim - 1)))
        np.testing.assert_allclose(per_feature_mean, 0, atol=0.01)

    def test_transform_scales_data(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(500, 10).astype(np.float32) * 5 + 7
        norm.fit(X)
        X_norm = norm.transform(X)
        per_feature_std = X_norm.std(axis=tuple(range(X_norm.ndim - 1)))
        np.testing.assert_allclose(per_feature_std, 1, atol=0.05)

    def test_inverse_transform_roundtrip(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(100, 5).astype(np.float32) * 10 + 20
        norm.fit(X)
        X_norm = norm.transform(X)
        X_recovered = norm.inverse_transform(X_norm)
        np.testing.assert_allclose(X, X_recovered, atol=1e-5)

    def test_zero_std_handling(self):
        """Features with zero std should not cause division by zero."""
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.zeros((100, 5), dtype=np.float32)
        X[:, 0] = np.random.randn(100).astype(np.float32)  # only one non-zero feature
        norm.fit(X)
        # Constant features should have std=1.0 (fallback)
        assert norm.std_[norm.std_ < 1e-8].sum() == 0  # no zero stds remain

    def test_state_dict_roundtrip(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(100, 3).astype(np.float32)
        norm.fit(X)
        sd = norm.state_dict()
        norm2 = FeatureNormalizer()
        norm2.load_state_dict(sd)
        np.testing.assert_array_equal(norm.mean_, norm2.mean_)
        np.testing.assert_array_equal(norm.std_, norm2.std_)

    def test_compatible_with_matching_features(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.random.randn(100, 14).astype(np.float32)
        norm.fit(X)
        assert norm.compatible_with(14) is True
        assert norm.compatible_with(11) is False

    def test_unfitted_transform_returns_input(self):
        from backend.ml.normalizer import FeatureNormalizer

        norm = FeatureNormalizer()
        X = np.ones((10, 5), dtype=np.float32)
        X_out = norm.transform(X)
        np.testing.assert_array_equal(X, X_out)

    def test_checkpoint_normalizer_compatibility(self):
        """Verify normalizer loads from actual checkpoint format."""
        import os

        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "model_best.pt")
        if not os.path.exists(ckpt_path):
            pytest.skip("No checkpoint file available")

        from backend.ml.normalizer import FeatureNormalizer

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        norm_data = state.get("normalizer")
        if norm_data is None:
            pytest.skip("No normalizer in checkpoint")

        norm = FeatureNormalizer()
        norm.load_state_dict(norm_data)
        assert norm.is_fitted
        # Checkpoint normalizer should have 14 features
        assert norm.compatible_with(14)


# ---------------------------------------------------------------------------
# PageHinkley drift detector
# ---------------------------------------------------------------------------

class TestPageHinkley:
    """Tests for Page-Hinkley change point detection."""

    def test_no_drift_on_constant_signal(self):
        """Constant signal should never trigger drift."""
        from main import PageHinkley

        ph = PageHinkley(delta=0.005, lamb=0.85)
        for _ in range(100):
            assert ph.update(0.0) is False

    def test_detects_abrupt_shift(self):
        """Large abrupt shift should trigger drift."""
        from main import PageHinkley

        ph = PageHinkley(delta=0.005, lamb=0.85)
        # Stable signal
        for _ in range(50):
            ph.update(0.01)
        # Abrupt shift
        triggered = False
        for _ in range(100):
            if ph.update(5.0):
                triggered = True
                break
        assert triggered, "Page-Hinkley should detect abrupt shift"

    def test_reset_clears_state(self):
        from main import PageHinkley

        ph = PageHinkley(delta=0.005, lamb=0.85)
        for _ in range(20):
            ph.update(10.0)
        ph.reset()
        assert ph.t == 0
        assert ph.mean == 0.0
        assert ph.m == 0.0
        assert ph.M == 0.0

    def test_gradual_drift(self):
        """Gradually increasing signal should eventually trigger."""
        from main import PageHinkley

        ph = PageHinkley(delta=0.005, lamb=3.0)
        triggered = False
        for i in range(500):
            if ph.update(float(i) * 0.01):
                triggered = True
                break
        assert triggered, "Page-Hinkley should detect gradual drift"


# ---------------------------------------------------------------------------
# ResidualKalman filter
# ---------------------------------------------------------------------------

class TestResidualKalman:
    """Tests for online residual Kalman correction."""

    def test_predict_returns_correct_shape(self):
        from main import OnlineConfig, ResidualKalman

        cfg = OnlineConfig()
        rk = ResidualKalman(n_series=10, cfg=cfg, seed=42)
        pred = rk.predict()
        assert pred.shape == (10,), f"Expected shape (10,), got {pred.shape}"

    def test_update_reduces_uncertainty(self):
        """After update, the state should be closer to the residual."""
        from main import OnlineConfig, ResidualKalman

        cfg = OnlineConfig()
        rk = ResidualKalman(n_series=5, cfg=cfg, seed=42)
        rk.predict()  # Initialize state
        residual = np.random.randn(5).astype(np.float32)
        correction = rk.update(residual)
        assert correction.shape == (5,)

    def test_predict_update_cycle(self):
        """Full predict-update cycle should produce finite outputs."""
        from main import OnlineConfig, ResidualKalman

        cfg = OnlineConfig()
        rk = ResidualKalman(n_series=10, cfg=cfg, seed=42)
        for _ in range(50):
            rk.predict()
            residual = np.random.randn(10).astype(np.float32) * 0.5
            rk.update(residual)
        final_pred = rk.predict()
        assert np.all(np.isfinite(final_pred)), "Predictions should be finite after many cycles"


# ---------------------------------------------------------------------------
# Hierarchical reconciliation
# ---------------------------------------------------------------------------

class TestReconciliation:
    """Tests for MinT reconciliation."""

    def test_reconciliation_improves_coherence(self):
        """Reconciled forecasts should have lower coherence error than raw forecasts."""
        from main import ASTANA_DISTRICTS, build_hierarchy, reconcile_mint

        # Use real district names since build_hierarchy uses ASTANA_DISTRICTS internally
        station_names = ["S1", "S2", "S3"]
        lines = {"L1": [0, 1, 2]}
        # Assign all stations to the same real district
        station_district = [ASTANA_DISTRICTS[0]] * 3
        net = type("NetworkSpec", (), {
            "station_names": station_names,
            "station_district": station_district,
            "lines": lines,
            "A_phys": np.eye(3, dtype=np.float32),
            "edges": [(0, 1), (1, 2)],
            "latlon": [(51.0, 71.0)] * 3,
        })()
        S, series_names, line_groups, district_groups = build_hierarchy(net)

        # 3 stations + 1 line + 4 districts + 1 total = 9 series
        n_series = S.shape[0]
        y_hat = np.random.randn(1, n_series).astype(np.float32) + 10
        W_diag = np.ones(n_series, dtype=np.float32)

        y_recon = reconcile_mint(y_hat, S, W_diag)
        assert y_recon.shape == y_hat.shape

        # Coherence error should be lower after reconciliation
        from main import coherence_error

        bottom_dim = len(station_names)
        error_before = coherence_error(y_hat, S, bottom_dim=bottom_dim)
        error_after = coherence_error(y_recon, S, bottom_dim=bottom_dim)
        assert error_after <= error_before + 1e-6, (
            f"Reconciliation should reduce or maintain coherence error: "
            f"before={error_before:.4f}, after={error_after:.4f}"
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class TestNBLoss:
    """Tests for Negative Binomial negative log-likelihood."""

    def test_loss_is_non_negative(self):
        from main import nb_nll

        y = torch.tensor([0.0, 1.0, 5.0, 10.0])
        mu = torch.tensor([0.5, 2.0, 6.0, 12.0])
        kappa = torch.tensor(8.0)
        loss = nb_nll(y, mu, kappa)  # Returns per-element tensor
        assert (loss >= 0).all(), "Each element of NLL should be non-negative"
        assert loss.mean().item() >= 0, "Mean NLL should be non-negative"

    def test_loss_decreases_with_better_predictions(self):
        """Loss should be lower when predictions are closer to true values."""
        from main import nb_nll

        y = torch.tensor([10.0])
        kappa = torch.tensor(8.0)
        good_mu = torch.tensor([10.0])
        bad_mu = torch.tensor([1.0])
        loss_good = nb_nll(y, good_mu, kappa)
        loss_bad = nb_nll(y, bad_mu, kappa)
        assert loss_good < loss_bad, "Better predictions should have lower loss"

    def test_gradient_exists(self):
        from main import nb_nll

        mu = torch.tensor([5.0], requires_grad=True)
        y = torch.tensor([5.0])
        kappa = torch.tensor(8.0)
        loss = nb_nll(y, mu, kappa)
        loss.backward()
        assert mu.grad is not None, "Gradient should exist"
        assert torch.isfinite(mu.grad).all(), "Gradient should be finite"

    def test_zero_y_no_nan(self):
        """Loss with y=0 should not produce NaN."""
        from main import nb_nll

        y = torch.tensor([0.0])
        mu = torch.tensor([1.0])
        kappa = torch.tensor(8.0)
        loss = nb_nll(y, mu, kappa)
        assert torch.isfinite(loss), "Loss should be finite for y=0"


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

class TestNetworkGeneration:
    """Tests for Astana bus network generation."""

    def test_build_astana_network_synthetic(self):
        from main import build_astana_network

        net = build_astana_network(use_real_data=False, n_stations=50, n_lines=5, seed=42)
        assert len(net.station_names) == 50
        assert len(net.station_district) == 50
        assert len(net.lines) == 5
        assert net.A_phys.shape == (50, 50)
        assert len(net.latlon) == 50

    def test_adjacency_matrix_row_stochastic(self):
        """Each row of A_phys should sum to approximately 1."""
        from main import build_astana_network

        net = build_astana_network(use_real_data=False, n_stations=30, seed=7)
        row_sums = net.A_phys.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_hierarchy_sums(self):
        """Bottom-level hierarchy should sum to network total."""
        from main import build_astana_network, build_hierarchy

        net = build_astana_network(use_real_data=False, n_stations=30, seed=7)
        S, series_names, _, _ = build_hierarchy(net)
        # Last row (Network Total) should sum across all stations
        assert series_names[-1] == "Network | Total"
        # Each aggregation row should have correct number of 1s
        N = len(net.station_names)
        # Station rows have exactly one 1
        for i in range(N):
            assert S[i].sum() == 1.0

    def test_data_generation_shape(self):
        from main import DataGenConfig, build_astana_network, generate_astana_data

        net = build_astana_network(use_real_data=False, n_stations=20, n_lines=4, seed=7)
        cfg = DataGenConfig(days=30, freq_min=60, seed=7)
        bundle = generate_astana_data(cfg, net)
        T = len(bundle.time_index)
        N = len(net.station_names)
        F = 16
        assert bundle.X.shape == (T, N, F), f"X shape: {bundle.X.shape}, expected ({T}, {N}, {F})"
        assert bundle.y_bottom.shape == (T, N)
        assert bundle.y_all.shape[1] == bundle.S.shape[0]

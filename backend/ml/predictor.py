"""Prediction engine — loads model artifact and generates multi-horizon forecasts.

Applies z-score normalization before inference when a FeatureNormalizer is
available in the checkpoint, ensuring predictions match training-time behavior.
"""

import logging
import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import torch

from backend.ml.artifact_store import get_production_artifact
from backend.ml.data_loader import build_adjacency, build_feature_tensor
from backend.ml.model import DTSGSSF
from backend.ml.normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)

# Module-level caches (thread-safe singletons)
_model_cache: dict[str, DTSGSSF] = {}
_normalizer_cache: dict[str, FeatureNormalizer] = {}
_cache_lock = threading.Lock()
_NOT_LOADED_SENTINEL = "NOT_LOADED"  # Marks versions whose artifact file is missing


def load_model(
    artifact_path: str,
    N: int,
    n_series: int,
    n_agg: int,
    A_phys: np.ndarray,
    device: str = "cpu",
) -> tuple[DTSGSSF, FeatureNormalizer | None]:
    """Load a trained model and its normalizer from an artifact path.

    Returns:
        Tuple of (model, normalizer). Normalizer is None if not found in checkpoint.
    """
    state = torch.load(artifact_path, map_location=device, weights_only=False)
    config = state.get("config", {})

    model = DTSGSSF(
        N=N,
        F_in=config.get("F_in", 16),
        n_series=n_series,
        n_agg=n_agg,
        A_phys=A_phys,
        d_model=config.get("d_model", 192),
        horizon=config.get("horizon", 4),
        K=config.get("K", 3),
        lora_r=config.get("lora_r", 16),
        dropout=config.get("dropout", 0.1),
        n_heads=config.get("n_heads", 6),
    )
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state)
    model.eval()

    # Load normalizer from checkpoint if available
    normalizer = None
    norm_state = state.get("normalizer")
    if norm_state is not None:
        try:
            normalizer = FeatureNormalizer()
            normalizer.load_state_dict(norm_state)
            f_in = config.get("F_in", 16)
            if normalizer.compatible_with(f_in):
                logger.info(
                    "Loaded feature normalizer from checkpoint (features=%d)",
                    normalizer.mean_.shape[-1],
                )
            else:
                logger.warning(
                    "Normalizer features (%d) != model F_in (%d); skipping normalization",
                    normalizer.mean_.shape[-1],
                    f_in,
                )
                normalizer = None
        except Exception as e:
            logger.warning("Failed to load normalizer from checkpoint: %s", e)
            normalizer = None
    else:
        logger.warning("No normalizer found in checkpoint; features will NOT be normalized")

    return model, normalizer


def get_cached_model(
    db_session=None,
) -> tuple[DTSGSSF | None, FeatureNormalizer | None]:
    """Get or load the production model and normalizer (cached across calls).

    Returns:
        Tuple of (model, normalizer). Either or both may be None on failure.
    """
    # Use provided session or create a short-lived one
    own_session = db_session is None
    if own_session:
        from backend.database import SessionLocal

        db_session = SessionLocal()

    try:
        artifact = get_production_artifact(db_session)
        if artifact is None:
            return None, None

        version = artifact.version

        # Fast path: return cached model or sentinel for previously-failed loads
        if version in _model_cache:
            cached = _model_cache[version]
            if cached is _NOT_LOADED_SENTINEL:
                return None, None  # Previously failed — skip retry
            normalizer = _normalizer_cache.get(version)
            return cached, normalizer

        with _cache_lock:
            if version in _model_cache:
                model = _model_cache[version]
                normalizer = _normalizer_cache.get(version)
                return model, normalizer

            A_phys, stop_ids, _station_idx = build_adjacency(db_session)
            N = len(stop_ids)

            # Read n_series/n_agg from checkpoint config, fallback to DB topology
            artifact_path = artifact.artifact_path
            try:
                ckpt = torch.load(artifact_path, map_location="cpu", weights_only=False)
                ckpt_config = ckpt.get("config", {})
            except Exception:
                ckpt_config = {}

            n_series = ckpt_config.get("n_series", 0)
            n_agg = ckpt_config.get("n_agg", 0)

            if n_series <= 0 or n_agg <= 0:
                # Fallback: compute from DB topology matching training hierarchy
                from backend.models_orm import RouteORM, StationORM

                n_routes = db_session.query(RouteORM).count()
                districts = {
                    s.district
                    for s in db_session.query(StationORM).all()
                    if s.district
                }
                n_districts = len(districts)
                n_series = N + n_routes + n_districts + 1  # +1 for network total
                n_agg = n_series - N
                logger.info(
                    "Checkpoint n_series/n_agg not found; computed from DB topology: "
                    "N=%d, n_routes=%d, n_districts=%d → n_series=%d, n_agg=%d",
                    N, n_routes, n_districts, n_series, n_agg,
                )
            else:
                logger.info(
                    "Using checkpoint n_series=%d, n_agg=%d (N=%d)",
                    n_series, n_agg, N,
                )
            model, normalizer = load_model(
                artifact_path=artifact.artifact_path,
                N=N,
                n_series=n_series,
                n_agg=n_agg,
                A_phys=A_phys,
                device="cpu",
            )

            # Clear old entries, keep only latest
            _model_cache.clear()
            _normalizer_cache.clear()
            if model is not None:
                _model_cache[version] = model
            if normalizer is not None:
                _normalizer_cache[version] = normalizer

            return model, normalizer
    except Exception as e:
        logger.error("Failed to load production model: %s", e)
        # Cache the failure so we don't retry on every call (e.g., every simulation tick)
        if artifact and version:
            with _cache_lock:
                if version not in _model_cache:
                    _model_cache[version] = _NOT_LOADED_SENTINEL
        return None, None
    finally:
        if own_session:
            db_session.close()


def generate_predictions(
    model: DTSGSSF,
    session,
    station_idx: dict[str, int],
    stop_ids: list[str],
    horizons: list[int] | None = None,
    normalizer: FeatureNormalizer | None = None,
) -> list[dict]:
    """Generate multi-horizon predictions using the loaded model.

    Args:
        model: Loaded DTSGSSF model instance.
        session: DB session for feature data.
        station_idx: Mapping of station IDs to indices.
        stop_ids: List of station IDs in order.
        horizons: Prediction horizons in minutes.
        normalizer: Feature normalizer fitted on training data. If provided,
                    features are z-score normalized before inference.
    """
    if horizons is None:
        horizons = [15, 30, 60, 120]
    now = datetime.now(UTC)
    predictions = []

    try:
        x, _ = build_feature_tensor(session, station_idx, stop_ids, now)

        # Apply z-score normalization before inference
        if normalizer is not None and normalizer.is_fitted:
            n_features_in = x.shape[-1]
            if normalizer.compatible_with(n_features_in):
                x = normalizer.transform(x)
            else:
                logger.warning(
                    "Normalizer features (%d) != input features (%d); "
                    "skipping normalization. Predictions may be inaccurate.",
                    normalizer.mean_.shape[-1],
                    n_features_in,
                )

        device = next(model.parameters()).device
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, kappa = model(x_tensor)
        mu_np = mu.cpu().numpy().squeeze()
        kappa_val = float(torch.clamp(kappa, min=0.01).cpu())

        len(stop_ids)
        for h_idx, horizon_min in enumerate(horizons):
            h_hours = horizon_min / 60.0
            step = min(h_idx + 1, mu_np.shape[0] - 1)
            ts = now + timedelta(minutes=horizon_min)
            for n_idx, sid in enumerate(stop_ids):
                if n_idx < mu_np.shape[1]:
                    pred_val = float(max(0, mu_np[step, n_idx]))
                    confidence = float(1.0 / (1.0 + kappa_val * h_hours))
                    predictions.append(
                        {
                            "station_id": sid,
                            "timestamp": ts.isoformat(),
                            "predicted": pred_val,
                            "confidence": confidence,
                            "horizon_minutes": horizon_min,
                            "model_version": "dts-gssf",
                        }
                    )
    except Exception as e:
        logger.error("Prediction generation failed: %s", e)

    return predictions


def generate_predictions_from_cache(session) -> list[dict]:
    """Generate predictions using the cached production model and normalizer."""
    model, normalizer = get_cached_model(session)
    if model is None:
        return []

    _A_phys, stop_ids, station_idx = build_adjacency(session)
    return generate_predictions(model, session, station_idx, stop_ids, normalizer=normalizer)


def generate_mock_predictions(
    stations: list[dict],
    horizons: list[int] | None = None,
) -> list[dict]:
    """Generate mock predictions when no trained model is available.

    Uses a rush-hour-aware Gaussian profile matching typical transit patterns
    instead of a flat sinusoidal baseline.
    """
    if horizons is None:
        horizons = [15, 30, 60, 120]
    now = datetime.now(UTC)
    predictions = []
    for station in stations:
        base = station.get("ridership_24h", 1500) / 24.0
        for horizon_min in horizons:
            h = now.hour + horizon_min / 60.0
            # Gaussian rush-hour profile matching training data patterns
            if 6 <= h <= 9:
                factor = 1.4 + 0.3 * np.cos(np.pi * (h - 7.5) / 2.5)
            elif 17 <= h <= 20:
                factor = 1.3 + 0.25 * np.cos(np.pi * (h - 18.5) / 2.5)
            elif 11 <= h <= 13:
                factor = 0.9 + 0.15 * np.cos(np.pi * (h - 12) / 1.5)
            elif 0 <= h < 6 or h > 22:
                factor = 0.15
            else:
                factor = 0.55
            noise = np.random.normal(0, base * 0.08)
            predicted = max(1, int(base * factor + noise))
            confidence = max(0.5, 0.95 - horizon_min / 600.0)
            predictions.append(
                {
                    "station_id": station["stop_id"],
                    "timestamp": (now + timedelta(minutes=horizon_min)).isoformat(),
                    "predicted": predicted,
                    "confidence": round(confidence, 3),
                    "horizon_minutes": horizon_min,
                    "model_version": "mock",
                }
            )
    return predictions

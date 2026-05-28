"""Prediction engine — loads model artifact and generates multi-horizon forecasts."""
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import torch

from backend.ml.model import DTSGSSF
from backend.ml.data_loader import build_adjacency, build_feature_tensor
from backend.ml.artifact_store import get_production_artifact

# Module-level model cache (thread-safe singleton)
_model_cache: Dict[str, DTSGSSF] = {}
_cache_lock = threading.Lock()


def load_model(artifact_path: str, N: int, n_series: int, n_agg: int, A_phys: np.ndarray,
               device: str = "cpu") -> DTSGSSF:
    """Load a trained model from artifact path, reading config from the saved artifact."""
    state = torch.load(artifact_path, map_location=device, weights_only=False)
    config = state.get("config", {})

    model = DTSGSSF(
        N=N,
        F_in=config.get("F_in", 11),
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
    return model


def get_cached_model() -> Optional[DTSGSSF]:
    """Get or load the production model (cached across calls)."""
    artifact = get_production_artifact()
    if artifact is None:
        return None

    version = artifact.version
    if version in _model_cache:
        return _model_cache[version]

    with _cache_lock:
        if version in _model_cache:
            return _model_cache[version]

        from backend.database import SessionLocal
        session = SessionLocal()
        try:
            A_phys, stop_ids, station_idx = build_adjacency(session)
            N = len(stop_ids)
            n_series = N
            n_agg = 3
            model = load_model(
                artifact_path=artifact.artifact_path,
                N=N, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
                device="cpu",
            )
            # Clear old entries, keep only latest
            _model_cache.clear()
            _model_cache[version] = model
            return model
        except Exception as e:
            print(f"Failed to load production model: {e}")
            return None
        finally:
            session.close()


def generate_predictions(
    model: DTSGSSF,
    session,
    station_idx: Dict[str, int],
    stop_ids: List[str],
    horizons: List[int] = [15, 30, 60, 120],
) -> List[Dict]:
    """Generate multi-horizon predictions using the loaded model."""
    now = datetime.now(timezone.utc)
    predictions = []

    try:
        x, _ = build_feature_tensor(session, station_idx, stop_ids, now)
        device = next(model.parameters()).device
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, kappa = model(x_tensor)
        mu_np = mu.cpu().numpy().squeeze()
        kappa_val = float(torch.clamp(kappa, min=0.01).cpu())

        N = len(stop_ids)
        for h_idx, horizon_min in enumerate(horizons):
            h_hours = horizon_min / 60.0
            step = min(h_idx + 1, mu_np.shape[0] - 1)
            ts = now + timedelta(minutes=horizon_min)
            for n_idx, sid in enumerate(stop_ids):
                if n_idx < mu_np.shape[1]:
                    pred_val = float(max(0, mu_np[step, n_idx]))
                    confidence = float(1.0 / (1.0 + kappa_val * h_hours))
                    predictions.append({
                        "station_id": sid,
                        "timestamp": ts.isoformat(),
                        "predicted": pred_val,
                        "confidence": confidence,
                        "horizon_minutes": horizon_min,
                        "model_version": "dts-gssf",
                    })
    except Exception as e:
        print(f"Prediction generation failed: {e}")

    return predictions


def generate_predictions_from_cache(session) -> List[Dict]:
    """Generate predictions using the cached production model."""
    model = get_cached_model()
    if model is None:
        return []

    A_phys, stop_ids, station_idx = build_adjacency(session)
    return generate_predictions(model, session, station_idx, stop_ids)


def generate_mock_predictions(
    stations: List[Dict],
    horizons: List[int] = [15, 30, 60, 120],
) -> List[Dict]:
    """Generate mock predictions when no trained model is available."""
    now = datetime.now(timezone.utc)
    predictions = []
    for station in stations:
        base_ridership = station.get("ridership_24h", 1500) / 24.0
        for horizon_min in horizons:
            h = now.hour + horizon_min / 60.0
            hour_factor = max(0.1, 0.3 + 0.7 * max(0, np.sin(np.pi * (h - 6) / 12))) if 6 <= h <= 22 else 0.1
            predicted = int(base_ridership * hour_factor + np.random.randint(-30, 30))
            confidence = max(0.5, 0.95 - horizon_min / 600.0)
            predictions.append({
                "station_id": station["stop_id"],
                "timestamp": (now + timedelta(minutes=horizon_min)).isoformat(),
                "predicted": max(0, predicted),
                "confidence": round(confidence, 3),
                "horizon_minutes": horizon_min,
                "model_version": "mock",
            })
    return predictions
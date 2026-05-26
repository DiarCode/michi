"""Prediction engine — loads model artifact and generates multi-horizon forecasts."""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import torch

from backend.ml.model import DTSGSSF
from backend.ml.data_loader import build_adjacency, build_feature_tensor


def load_model(artifact_path: str, N: int, n_series: int, n_agg: int, A_phys: np.ndarray,
              device: str = "cpu") -> DTSGSSF:
    """Load a trained model from artifact path."""
    model = DTSGSSF(
        N=N, F_in=11, n_series=n_series, n_agg=n_agg, A_phys=A_phys,
        d_model=128, horizon=4, K=3, lora_r=8,
    )
    state = torch.load(artifact_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model


def generate_predictions(
    model: DTSGSSF,
    session,
    station_idx: Dict[str, int],
    stop_ids: List[str],
    horizons: List[int] = [15, 30, 60, 120],
) -> List[Dict]:
    """Generate multi-horizon predictions using the loaded model.

    Returns list of prediction dicts suitable for storing in forecasts table.
    """
    now = datetime.now(timezone.utc)
    predictions = []

    try:
        x, _ = build_feature_tensor(session, station_idx, stop_ids, now)
        device = next(model.parameters()).device
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, kappa = model(x_tensor)
        mu_np = mu.cpu().numpy().squeeze()  # (H, n_series)
        kappa_val = float(torch.clamp(kappa, min=0.01).cpu())

        N = len(stop_ids)
        for h_idx, horizon_min in enumerate(horizons):
            h_hours = horizon_min / 60.0
            # Map horizon index to model output
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

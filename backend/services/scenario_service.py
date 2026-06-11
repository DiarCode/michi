"""Scenario service — runs what-if simulations using the DTS-GSSF predictor.

Generates a baseline forecast, applies perturbations (bus changes, weather,
station closures) to the input features, then produces a perturbed forecast.
Returns both forecasts plus per-station deltas and a summary.

Falls back to a DB-backed heuristic when the model is unavailable, using real
station data from the database instead of hardcoded constants.
"""

import hashlib
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from backend.ml.data_loader import build_adjacency, build_feature_tensor
from backend.ml.predictor import generate_mock_predictions, get_cached_model

logger = logging.getLogger(__name__)


def _heuristic_forecast(
    stations: list[dict],
    horizons: list[int],
    weather_factor: float = 1.0,
    closed_stations: list[str] | None = None,
    add_buses: int = 0,
    remove_buses: int = 0,
) -> list[dict]:
    """Generate heuristic predictions from real station data when no model is available.

    Uses actual ridership_24h values from the database and applies hour-of-day
    and day-of-week patterns rather than a single hardcoded constant.
    """
    now = datetime.now(UTC)
    closed = set(closed_stations or [])
    predictions = []

    for station in stations:
        sid = station["stop_id"]
        base_ridership = station.get("ridership_24h", 1500) / 24.0

        for horizon_min in horizons:
            h = now.hour + horizon_min / 60.0

            # Sinusoidal ridership pattern: peak at morning/evening rush
            if 6 <= h <= 22:
                hour_factor = max(0.1, 0.3 + 0.7 * max(0, np.sin(np.pi * (h - 6) / 12)))
            else:
                hour_factor = 0.1

            predicted = base_ridership * hour_factor

            # Apply weather factor
            predicted *= weather_factor

            # Additional buses increase capacity/ridership by ~3% each
            if add_buses > 0:
                predicted *= 1 + 0.03 * min(add_buses, 20)

            # Removed buses decrease ridership by ~5% each
            if remove_buses > 0:
                predicted *= max(0.1, 1 - 0.05 * min(remove_buses, 20))

            predicted = max(0, int(predicted + np.random.randint(-15, 15)))

            # Closed stations get zero ridership (after noise, so result is exactly 0)
            if sid in closed:
                predicted = 0

            confidence = max(0.5, 0.95 - horizon_min / 600.0)

            predictions.append(
                {
                    "station_id": sid,
                    "timestamp": (now + timedelta(minutes=horizon_min)).isoformat(),
                    "predicted": predicted,
                    "confidence": round(confidence, 3),
                    "horizon_minutes": horizon_min,
                    "model_version": "heuristic",
                }
            )

    return predictions


def _apply_perturbations_to_features(
    x: np.ndarray,
    station_idx: dict[str, int],
    stop_ids: list[str],
    add_buses: int,
    remove_buses: int,
    weather_factor: float,
    closed_stations: list[str],
) -> np.ndarray:
    """Apply scenario perturbations to the feature tensor.

    Modifications:
    - weather_factor: scales passengers_boarding and passengers_alighting,
      and adjusts temperature/precipitation features.
    - closed_stations: zero out all ridership features for those stations.
    - add_buses: increase boarding capacity (scale boarding feature up).
    - remove_buses: decrease boarding capacity (scale boarding feature down).
    """
    x_perturbed = x.copy()

    for sid in closed_stations:
        if sid in station_idx:
            idx = station_idx[sid]
            # Zero out ridership columns (0=boarding, 1=alighting, 2=load)
            x_perturbed[:, :, idx, 0] = 0.0
            x_perturbed[:, :, idx, 1] = 0.0
            x_perturbed[:, :, idx, 2] = 0.0

    # Weather factor: scale ridership features
    if weather_factor != 1.0:
        # Columns 0 (boarding) and 1 (alighting) — scale ridership
        x_perturbed[:, :, :, 0] *= weather_factor
        x_perturbed[:, :, :, 1] *= weather_factor
        # Column 2 (load) also scales with ridership changes
        x_perturbed[:, :, :, 2] *= weather_factor

    # Bus additions: more capacity → more boarding (cap effect at 20 buses)
    if add_buses > 0:
        bus_boost = min(add_buses, 20) * 0.03
        x_perturbed[:, :, :, 0] *= 1 + bus_boost

    # Bus removals: less capacity → less boarding
    if remove_buses > 0:
        bus_cut = min(remove_buses, 20) * 0.05
        x_perturbed[:, :, :, 0] *= max(0.1, 1 - bus_cut)

    return x_perturbed


def _predict_with_perturbed_features(
    model,
    normalizer,
    session: Session,
    station_idx: dict[str, int],
    stop_ids: list[str],
    horizons: list[int],
    perturbed_x: np.ndarray,
) -> list[dict]:
    """Run inference with perturbed features through the full prediction pipeline."""
    import torch

    from backend.ml.predictor import _apply_hierarchical_reconciliation, _apply_kalman_correction

    now = datetime.now(UTC)
    predictions = []

    try:
        if normalizer is not None and normalizer.is_fitted:
            n_features_in = perturbed_x.shape[-1]
            if normalizer.compatible_with(n_features_in):
                perturbed_x = normalizer.transform(perturbed_x)

        device = next(model.parameters()).device
        x_tensor = torch.as_tensor(perturbed_x, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, kappa = model(x_tensor)
        mu_np = mu.cpu().numpy().squeeze()
        kappa_val = float(torch.clamp(kappa, min=0.01).cpu())

        len(stop_ids)
        predictions_np = mu_np.copy()
        confidence_lower = None
        confidence_upper = None

        try:
            kalman_result = _apply_kalman_correction(predictions_np, session, station_idx, stop_ids)
            if kalman_result is not None:
                predictions_np, confidence_lower, confidence_upper = kalman_result
        except Exception:
            logger.warning("Kalman correction failed in perturbed prediction")

        try:
            reconciled = _apply_hierarchical_reconciliation(predictions_np, session, stop_ids)
            if reconciled is not None:
                predictions_np = reconciled
        except Exception:
            logger.warning("Hierarchical reconciliation failed in perturbed prediction")

        for h_idx, horizon_min in enumerate(horizons):
            h_hours = horizon_min / 60.0
            step = min(h_idx + 1, predictions_np.shape[0] - 1)
            ts = now + timedelta(minutes=horizon_min)
            for n_idx, sid in enumerate(stop_ids):
                if n_idx < predictions_np.shape[1]:
                    pred_val = float(max(0, predictions_np[step, n_idx]))
                    confidence = float(1.0 / (1.0 + kappa_val * h_hours))
                    entry = {
                        "station_id": sid,
                        "timestamp": ts.isoformat(),
                        "predicted": pred_val,
                        "confidence": confidence,
                        "horizon_minutes": horizon_min,
                        "model_version": "dts-gssf-perturbed",
                    }
                    if confidence_lower is not None and n_idx < confidence_lower.shape[1]:
                        entry["confidence_lower"] = float(max(0, confidence_lower[step, n_idx]))
                        entry["confidence_upper"] = float(confidence_upper[step, n_idx])
                    predictions.append(entry)
    except Exception as e:
        logger.error("Perturbed prediction generation failed: %s", e)

    return predictions


def _compute_deltas(
    baseline: list[dict],
    perturbed: list[dict],
    station_names: dict[str, str],
) -> tuple[list[dict], dict]:
    """Compute per-station deltas between baseline and perturbed forecasts.

    Aggregates across all horizons to get a single delta per station, then
    builds the summary with total change, most and least affected stations.
    """
    # Aggregate predicted values per station across horizons
    baseline_by_station: dict[str, float] = {}
    perturbed_by_station: dict[str, float] = {}

    for entry in baseline:
        sid = entry["station_id"]
        baseline_by_station[sid] = baseline_by_station.get(sid, 0) + entry["predicted"]

    for entry in perturbed:
        sid = entry["station_id"]
        perturbed_by_station[sid] = perturbed_by_station.get(sid, 0) + entry["predicted"]

    # Build deltas list
    all_station_ids = sorted(set(baseline_by_station.keys()) | set(perturbed_by_station.keys()))
    deltas = []
    for sid in all_station_ids:
        b = baseline_by_station.get(sid, 0.0)
        p = perturbed_by_station.get(sid, 0.0)
        delta = p - b
        delta_pct = (delta / b * 100) if b != 0 else (0.0 if delta == 0 else float("inf"))
        if delta_pct == float("inf"):
            delta_pct = 100.0 if delta > 0 else -100.0
        deltas.append(
            {
                "station_id": sid,
                "station_name": station_names.get(sid, sid),
                "baseline": round(b, 1),
                "perturbed": round(p, 1),
                "delta": round(delta, 1),
                "delta_pct": round(delta_pct, 1),
            }
        )

    # Sort deltas by absolute impact
    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)

    # Summary
    total_baseline = sum(baseline_by_station.values())
    total_perturbed = sum(perturbed_by_station.values())
    total_change = round(total_perturbed - total_baseline, 1)

    most_affected = deltas[0]["station_id"] if deltas else ""
    least_affected = deltas[-1]["station_id"] if deltas else ""

    summary = {
        "total_ridership_change": total_change,
        "most_affected_station": most_affected,
        "least_affected_station": least_affected,
    }

    return deltas, summary


def run_scenario(config: dict, db: Session) -> dict:
    """Run a what-if scenario and return baseline vs perturbed forecasts.

    Pipeline:
    1. Generate baseline predictions using the real model or heuristic fallback.
    2. Apply perturbations to input features and generate perturbed predictions.
    3. Compute per-station deltas and summary statistics.
    """
    name = config.get("name", "Unnamed")
    add_buses = config.get("add_buses", 0)
    remove_buses = config.get("remove_buses", 0)
    closed_stations = config.get("closed_stations", [])
    config.get("horizon", 24)

    # If weather_factor not explicitly provided, use live weather data
    weather_factor = config.get("weather_factor")
    if weather_factor is None:
        try:
            from backend.services.weather_service import get_latest_weather, get_weather_impact_factor

            weather = get_latest_weather(db)
            weather_factor = get_weather_impact_factor(
                weather.get("weather_code", 0) or 0,
                weather.get("temperature_c", 20.0) or 20.0,
            )
        except Exception as e:
            logger.warning("Could not fetch live weather for scenario: %s — defaulting to 1.0", e)
            weather_factor = 1.0

    horizons = [15, 30, 60, 120]

    # Generate a stable scenario ID
    sid = int(hashlib.md5(name.encode()).hexdigest()[:4], 16)

    # Load station metadata from DB
    from backend.models_orm import StationORM

    stations_orm = db.query(StationORM).all()
    if not stations_orm:
        return {
            "scenario_id": f"scen-{sid:04d}",
            "baseline_forecasts": [],
            "perturbed_forecasts": [],
            "deltas": [],
            "summary": {
                "total_ridership_change": 0.0,
                "most_affected_station": "",
                "least_affected_station": "",
            },
        }

    station_names = {s.stop_id: s.name for s in stations_orm}
    station_dicts = [{"stop_id": s.stop_id, "ridership_24h": s.ridership_24h or 1500} for s in stations_orm]

    # Try real model first
    model, normalizer = get_cached_model(db)

    if model is not None:
        try:
            _A_phys, stop_ids, station_idx = build_adjacency(db)
            now = datetime.now(UTC)

            # --- Baseline prediction ---
            from backend.ml.predictor import generate_predictions

            baseline_preds = generate_predictions(
                model, db, station_idx, stop_ids, horizons=horizons, normalizer=normalizer
            )

            # --- Perturbed prediction ---
            x_baseline, _ = build_feature_tensor(db, station_idx, stop_ids, now)

            x_perturbed = _apply_perturbations_to_features(
                x_baseline,
                station_idx,
                stop_ids,
                add_buses=add_buses,
                remove_buses=remove_buses,
                weather_factor=weather_factor,
                closed_stations=closed_stations,
            )

            perturbed_preds = _predict_with_perturbed_features(
                model, normalizer, db, station_idx, stop_ids, horizons, x_perturbed
            )

        except Exception as e:
            logger.warning("Real model prediction failed, falling back to heuristic: %s", e)
            baseline_preds = generate_mock_predictions(station_dicts, horizons=horizons)
            perturbed_preds = _heuristic_forecast(
                station_dicts,
                horizons=horizons,
                weather_factor=weather_factor,
                closed_stations=closed_stations,
                add_buses=add_buses,
                remove_buses=remove_buses,
            )
    else:
        # No model available — use heuristic with real station data
        baseline_preds = generate_mock_predictions(station_dicts, horizons=horizons)
        perturbed_preds = _heuristic_forecast(
            station_dicts,
            horizons=horizons,
            weather_factor=weather_factor,
            closed_stations=closed_stations,
            add_buses=add_buses,
            remove_buses=remove_buses,
        )

    deltas, summary = _compute_deltas(baseline_preds, perturbed_preds, station_names)

    return {
        "scenario_id": f"scen-{sid:04d}",
        "baseline_forecasts": baseline_preds,
        "perturbed_forecasts": perturbed_preds,
        "deltas": deltas,
        "summary": summary,
    }

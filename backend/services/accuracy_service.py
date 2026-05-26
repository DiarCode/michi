"""Prediction accuracy tracking — compares forecasts with actuals and stores metrics."""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
from backend.database import SessionLocal
from backend.models_orm import ForecastORM, HistoricalRidershipORM, PredictionAccuracyORM, ModelArtifactORM


def evaluate_accuracy(model_version: Optional[str] = None, hours_back: int = 24) -> Dict:
    """Compare past predictions with actual ridership and compute accuracy metrics."""
    session = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours_back)

        forecasts = (session.query(ForecastORM)
                     .filter(ForecastORM.timestamp >= cutoff)
                     .filter(ForecastORM.timestamp <= now)
                     .order_by(ForecastORM.timestamp).all())

        if not forecasts:
            return {"status": "no_data", "mae": None, "rmse": None, "mape": None, "count": 0}

        errors = []
        for fc in forecasts:
            actual = (session.query(HistoricalRidershipORM)
                     .filter(HistoricalRidershipORM.station_id == fc.station_id)
                     .filter(HistoricalRidershipORM.timestamp >= fc.timestamp.replace(tzinfo=timezone.utc) - timedelta(minutes=30))
                     .filter(HistoricalRidershipORM.timestamp <= fc.timestamp.replace(tzinfo=timezone.utc) + timedelta(minutes=30))
                     .first())
            if actual:
                pred = fc.predicted
                act = actual.passengers_boarding
                abs_err = abs(pred - act)
                pct_err = abs_err / max(act, 1)
                errors.append({
                    "station_id": fc.station_id,
                    "route_id": fc.route_id or "",
                    "forecast_timestamp": fc.timestamp.isoformat() if fc.timestamp else "",
                    "horizon_minutes": fc.horizon_minutes or 60,
                    "predicted": pred,
                    "actual": act,
                    "absolute_error": abs_err,
                    "mape": pct_err,
                })
                # Store individual accuracy record
                acc = PredictionAccuracyORM(
                    model_version=fc.model_version or model_version or "unknown",
                    station_id=fc.station_id,
                    route_id=fc.route_id or "",
                    forecast_timestamp=fc.timestamp.replace(tzinfo=timezone.utc) if fc.timestamp else now,
                    horizon_minutes=fc.horizon_minutes or 60,
                    predicted=pred,
                    actual=act,
                    absolute_error=abs_err,
                    mape=pct_err,
                    evaluated_at=now,
                )
                session.add(acc)

        session.commit()

        if not errors:
            return {"status": "no_actuals", "mae": None, "rmse": None, "mape": None, "count": 0}

        abs_errors = [e["absolute_error"] for e in errors]
        mape_vals = [e["mape"] for e in errors]

        return {
            "status": "ok",
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(np.array(abs_errors) ** 2))),
            "mape": float(np.mean(mape_vals)) * 100,
            "count": len(errors),
            "by_horizon": _group_by_horizon(errors),
        }
    finally:
        session.close()


def _group_by_horizon(errors: List[Dict]) -> Dict:
    """Group accuracy metrics by prediction horizon."""
    by_horizon = {}
    for e in errors:
        h = e["horizon_minutes"]
        by_horizon.setdefault(h, {"abs_errors": [], "mape_vals": []})
        by_horizon[h]["abs_errors"].append(e["absolute_error"])
        by_horizon[h]["mape_vals"].append(e["mape"])
    return {
        str(h): {
            "mae": float(np.mean(v["abs_errors"])),
            "mape": float(np.mean(v["mape_vals"])) * 100,
            "count": len(v["abs_errors"]),
        }
        for h, v in by_horizon.items()
    }


def get_accuracy_trend(days: int = 30) -> List[Dict]:
    """Get daily accuracy trend for the dashboard."""
    session = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        records = (session.query(PredictionAccuracyORM)
                   .filter(PredictionAccuracyORM.evaluated_at >= cutoff)
                   .order_by(PredictionAccuracyORM.evaluated_at).all())

        daily = {}
        for r in records:
            day = r.evaluated_at.strftime("%Y-%m-%d") if r.evaluated_at else "unknown"
            daily.setdefault(day, {"mae": [], "mape": [], "count": 0})
            if r.absolute_error is not None:
                daily[day]["mae"].append(r.absolute_error)
            if r.mape is not None:
                daily[day]["mape"].append(r.mape)
            daily[day]["count"] += 1

        return [
            {
                "date": day,
                "mae": float(np.mean(v["mae"])) if v["mae"] else None,
                "mape": float(np.mean(v["mape"])) * 100 if v["mape"] else None,
                "count": v["count"],
            }
            for day, v in sorted(daily.items())
        ]
    finally:
        session.close()

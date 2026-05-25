"""Forecast service - generates and retrieves ridership forecasts."""
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

MOCK_FORECAST: Dict[str, List[dict]] = {}


def generate_24h_forecast(station_id: str, base_ridership: int = 1000) -> List[dict]:
    """Generate a 24-hour forecast for a station."""
    now = datetime.now(timezone.utc)
    hourly = []
    for h in range(24):
        ts = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if h < now.hour:
            ts += timedelta(days=1)
        factor = 0.3 + 0.7 * max(0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 22 else 0.1
        predicted = int(base_ridership * factor + np.random.randint(-50, 50))
        confidence = round(0.85 + np.random.random() * 0.12, 3)
        hourly.append({
            "station_id": station_id,
            "timestamp": ts.isoformat(),
            "predicted": max(0, predicted),
            "confidence": confidence,
        })
    return hourly


def get_forecast(station_id: str) -> List[dict]:
    if station_id not in MOCK_FORECAST:
        MOCK_FORECAST[station_id] = generate_24h_forecast(station_id)
    return MOCK_FORECAST[station_id]


def get_kpi_metrics(db=None) -> dict:
    """Return KPI metrics. Queries DB when available, falls back to defaults."""
    defaults = {
        "total_stations": 0,
        "active_routes": 0,
        "avg_ridership": 0.0,
        "alerts_today": 0,
        "on_time_performance": 94.2,
        "peak_hour": "08:00",
    }
    if db is None:
        return defaults

    try:
        from backend.models_orm import StationORM, RouteORM, AlertORM, RidershipORM
        from sqlalchemy import func

        total_stations = db.query(StationORM).count()
        active_routes = db.query(RouteORM).count()

        avg_result = db.query(func.avg(StationORM.ridership_24h)).scalar()
        avg_ridership = round(float(avg_result), 1) if avg_result else 0.0

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        alerts_today = db.query(AlertORM).filter(AlertORM.created_at >= today_start).count()

        peak_hour_result = (db.query(RidershipORM.timestamp, func.sum(RidershipORM.passengers).label("total"))
                           .group_by(RidershipORM.timestamp)
                           .order_by(func.sum(RidershipORM.passengers).desc())
                           .first())
        peak_hour = peak_hour_result[0].strftime("%H:00") if peak_hour_result else "08:00"

        return {
            "total_stations": total_stations or defaults["total_stations"],
            "active_routes": active_routes or defaults["active_routes"],
            "avg_ridership": avg_ridership or defaults["avg_ridership"],
            "alerts_today": alerts_today or defaults["alerts_today"],
            "on_time_performance": defaults["on_time_performance"],
            "peak_hour": peak_hour,
        }
    except Exception:
        return defaults
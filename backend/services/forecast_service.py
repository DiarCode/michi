"""Forecast service — generates ridership forecasts using DTS-GSSF model with mock fallback."""
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.ml.predictor import generate_predictions_from_cache, generate_mock_predictions


def generate_24h_forecast(station_id: str, base_ridership: int = 1000) -> List[dict]:
    """Generate a 24-hour forecast for a station using the real model if available."""
    from backend.database import SessionLocal
    session = SessionLocal()
    try:
        predictions = generate_predictions_from_cache(session)
        if predictions:
            station_preds = [p for p in predictions if p["station_id"] == station_id]
            if station_preds:
                return station_preds
        from backend.models_orm import StationORM
        station = session.query(StationORM).filter(StationORM.stop_id == station_id).first()
        stations = [{"stop_id": station_id, "ridership_24h": base_ridership}]
        if station:
            stations = [{"stop_id": station.stop_id, "ridership_24h": station.ridership_24h or base_ridership}]
        return generate_mock_predictions(stations)
    finally:
        session.close()


def generate_all_forecasts(db) -> List[dict]:
    """Generate forecasts for all stations using the real model if available."""
    predictions = generate_predictions_from_cache(db)
    if predictions:
        return predictions
    from backend.models_orm import StationORM
    stations = db.query(StationORM).all()
    if not stations:
        return []
    station_dicts = [{"stop_id": s.stop_id, "ridership_24h": s.ridership_24h or 1000} for s in stations]
    return generate_mock_predictions(station_dicts)


def get_forecast(station_id: str) -> List[dict]:
    """Get cached or generate forecast for a station."""
    return generate_24h_forecast(station_id)


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
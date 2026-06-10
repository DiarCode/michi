"""Forecast service — generates ridership forecasts using DTS-GSSF model with mock fallback."""
import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.ml.predictor import generate_mock_predictions, generate_predictions_from_cache

logger = logging.getLogger(__name__)


def generate_24h_forecast(station_id: str, db: Session, base_ridership: int = 1000) -> list[dict]:
    """Generate a 24-hour forecast for a station using the real model if available."""
    predictions = generate_predictions_from_cache(db)
    if predictions:
        station_preds = [p for p in predictions if p["station_id"] == station_id]
        if station_preds:
            return station_preds
    from backend.models_orm import StationORM
    station = db.query(StationORM).filter(StationORM.stop_id == station_id).first()
    stations = [{"stop_id": station_id, "ridership_24h": base_ridership}]
    if station:
        stations = [{"stop_id": station.stop_id, "ridership_24h": station.ridership_24h or base_ridership}]
    return generate_mock_predictions(stations)


def generate_all_forecasts(db: Session) -> list[dict]:
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


def get_forecast(station_id: str, db: Session) -> list[dict]:
    """Get cached or generate forecast for a station."""
    return generate_24h_forecast(station_id, db=db)


def get_kpi_metrics(db: Session | None = None) -> dict:
    """Return KPI metrics. Queries DB when available, falls back to zero defaults."""
    defaults = {
        "total_stations": 0,
        "active_routes": 0,
        "avg_ridership": 0.0,
        "alerts_today": 0,
        "on_time_performance": 0.0,
        "peak_hour": "08:00",
    }
    if db is None:
        return defaults

    try:
        from sqlalchemy import func

        from backend.models_orm import AlertORM, PredictionAccuracyORM, RidershipORM, RouteORM, StationORM

        total_stations = db.query(StationORM).count()
        active_routes = db.query(RouteORM).count()

        avg_result = db.query(func.avg(StationORM.ridership_24h)).scalar()
        avg_ridership = round(float(avg_result), 1) if avg_result else 0.0

        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        alerts_today = db.query(AlertORM).filter(AlertORM.created_at >= today_start).count()

        peak_hour_result = (
            db.query(RidershipORM.timestamp, func.sum(RidershipORM.passengers).label("total"))
            .group_by(RidershipORM.timestamp)
            .order_by(func.sum(RidershipORM.passengers).desc())
            .first()
        )
        peak_hour = peak_hour_result[0].strftime("%H:00") if peak_hour_result else "08:00"

        # Compute on-time performance from prediction accuracy data if available
        on_time_performance = 0.0
        avg_mape_result = db.query(func.avg(PredictionAccuracyORM.mape)).filter(
            PredictionAccuracyORM.mape.isnot(None)
        ).scalar()
        if avg_mape_result is not None and avg_mape_result > 0:
            # Convert MAPE to on-time performance: 100 - MAPE (capped at 0)
            on_time_performance = round(max(0.0, 100.0 - float(avg_mape_result)), 1)

        return {
            "total_stations": total_stations or defaults["total_stations"],
            "active_routes": active_routes or defaults["active_routes"],
            "avg_ridership": avg_ridership or defaults["avg_ridership"],
            "alerts_today": alerts_today or defaults["alerts_today"],
            "on_time_performance": on_time_performance,
            "peak_hour": peak_hour,
        }
    except Exception as e:
        logger.warning("KPI metrics query failed: %s", e)
        return defaults

"""Passenger info API — crowding predictions, service changes, public messaging."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models_orm import AlertORM, ForecastORM, StationORM

router = APIRouter()


@router.get("/crowding")
def get_crowding_predictions(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """Get current and predicted crowding levels for all stations."""
    stations = db.query(StationORM).offset(offset).limit(limit).all()
    result = []
    datetime.now(UTC)

    for station in stations:
        forecasts = (
            db.query(ForecastORM)
            .filter(ForecastORM.station_id == station.stop_id)
            .order_by(ForecastORM.timestamp)
            .all()
        )

        crowding_levels = []
        for fc in forecasts[:4]:  # 4 horizons: 15, 30, 60, 120 min
            base = max(station.ridership_24h or 1500, 1) / 24
            level = (
                "low"
                if fc.predicted < base * 0.6
                else "medium"
                if fc.predicted < base * 1.2
                else "high"
                if fc.predicted < base * 1.8
                else "very_high"
            )
            crowding_levels.append(
                {
                    "horizon_minutes": fc.horizon_minutes or 60,
                    "predicted": int(fc.predicted),
                    "confidence": fc.confidence,
                    "level": level,
                }
            )

        result.append(
            {
                "station_id": station.stop_id,
                "name": station.name,
                "district": station.district,
                "current_crowding": crowding_levels[0]["level"] if crowding_levels else "unknown",
                "predictions": crowding_levels,
            }
        )

    return {"stations": result}


@router.get("/service-changes")
def get_service_changes(db: Session = Depends(get_db_session)):
    """Get approved service changes visible to passengers."""
    interventions = (
        db.query(AlertORM)
        .filter(AlertORM.severity.in_(["high", "critical"]))
        .order_by(AlertORM.created_at.desc())
        .limit(10)
        .all()
    )

    changes = []
    for alert in interventions:
        changes.append(
            {
                "id": alert.id,
                "title": alert.title,
                "message": alert.message or alert.what or "",
                "severity": alert.severity,
                "route_id": alert.route_id,
                "station_id": alert.station_id,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
            }
        )

    return {"service_changes": changes}


@router.get("/messaging-templates")
def get_messaging_templates():
    """Pre-built passenger messaging templates for common scenarios."""
    return {
        "templates": [
            {
                "id": "overcrowding",
                "title": "Station Overcrowding Alert",
                "body": "High passenger volume expected at {station} in the next {time} minutes. Consider alternative routes.",
            },
            {
                "id": "delay",
                "title": "Service Delay Notice",
                "body": "Route {route} is experiencing delays of approximately {minutes} min. We apologize for the inconvenience.",
            },
            {
                "id": "reroute",
                "title": "Service Reroute",
                "body": "Route {route} is temporarily rerouted via {alternative}. Please check the updated route map.",
            },
            {
                "id": "event",
                "title": "Event Travel Advisory",
                "body": "Large event at {venue} may cause increased wait times on routes {routes}. Allow extra travel time.",
            },
            {
                "id": "weather",
                "title": "Weather Advisory",
                "body": "Due to {weather_condition}, increased demand is expected. Additional buses are being deployed on high-demand routes.",
            },
        ]
    }

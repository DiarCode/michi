from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database import get_db_session
from backend.models_orm import StationORM, RouteORM, RouteStopORM, AlertORM
from backend.services.forecast_service import get_forecast
from backend.models import StationListResponse, StationDetailResponse, ForecastResponse

router = APIRouter()

STATION_CAPACITY = 3000  # Estimated max capacity per station


def _get_stations(db: Session):
    """Shared helper: return stations list from the database."""
    db_stations = db.query(StationORM).all()
    if db_stations:
        return [
            {"id": s.stop_id, "name": s.name, "lat": s.lat, "lon": s.lon,
             "district": s.district, "ridership_24h": s.ridership_24h}
            for s in db_stations
        ]
    return []


def _calc_load_pct(ridership_24h: int, hour: int) -> int:
    """Estimate load percentage for a given hour based on 24h ridership."""
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return min(95, int(ridership_24h / STATION_CAPACITY * 100 * 1.4))
    elif 6 <= hour <= 22:
        return min(70, int(ridership_24h / STATION_CAPACITY * 100 * 0.8))
    else:
        return min(30, int(ridership_24h / STATION_CAPACITY * 100 * 0.25))


@router.get("", response_model=StationListResponse)
def list_stations(hour: Optional[int] = Query(None, ge=0, le=23), db: Session = Depends(get_db_session)):
    """List stations, optionally with heatmap load data for a specific hour."""
    stations = _get_stations(db)

    if hour is not None:
        for s in stations:
            ridership = s.get("ridership_24h", 0) or 0
            s["load_percent"] = _calc_load_pct(ridership, hour)

    return {"stations": stations}


@router.get("/{station_id}/forecast", response_model=ForecastResponse)
def get_station_forecast(station_id: str, db: Session = Depends(get_db_session)):
    forecast = get_forecast(station_id, db=db)
    return {"station_id": station_id, "forecast": forecast}


@router.get("/{station_id}/detail", response_model=StationDetailResponse)
def get_station_detail(station_id: str, db: Session = Depends(get_db_session)):
    """Station detail with forecasts, connected routes, and active alerts."""
    station = db.query(StationORM).filter(StationORM.stop_id == station_id).first()

    station_info = None
    if station:
        station_info = {"id": station.stop_id, "name": station.name, "lat": station.lat,
                        "lon": station.lon, "district": station.district, "ridership_24h": station.ridership_24h}
    if not station_info:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    # Connected routes
    connected_routes = []
    route_stops = db.query(RouteStopORM).filter(RouteStopORM.station_id == station_id).all()
    for rs in route_stops:
        route = db.query(RouteORM).filter(RouteORM.route_id == rs.route_id).first()
        if route:
            connected_routes.append({"id": route.route_id, "name": route.name, "color": route.color})

    # Forecast
    forecast = get_forecast(station_id, db=db)

    # Alerts for this station
    alerts = db.query(AlertORM).filter(AlertORM.station_id == station_id).all()
    station_alerts = [{"severity": a.severity, "title": a.title, "message": a.message} for a in alerts]

    # Hourly ridership pattern
    hourly = [{"hour": i, "ridership": f["predicted"]} for i, f in enumerate(forecast)]

    return {
        "station": station_info,
        "connected_routes": connected_routes,
        "forecast": forecast,
        "alerts": station_alerts,
        "hourly_ridership": hourly,
    }
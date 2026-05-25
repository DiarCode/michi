from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models_orm import StationORM
from backend.services.forecast_service import get_forecast

router = APIRouter()

MOCK_STATIONS = [
    {"id": "S001", "name": "Nurly Zhol Station", "lat": 51.1605, "lon": 71.4704, "district": "Esil", "ridership_24h": 1840},
    {"id": "S002", "name": "Khan Shatyr", "lat": 51.1334, "lon": 71.4244, "district": "Esil", "ridership_24h": 3200},
    {"id": "S003", "name": "Bayterek", "lat": 51.1283, "lon": 71.4305, "district": "Esil", "ridership_24h": 4100},
    {"id": "S004", "name": "Astana Arena", "lat": 51.1081, "lon": 71.4024, "district": "Saryarka", "ridership_24h": 1500},
    {"id": "S005", "name": "Nazarbayev University", "lat": 51.0906, "lon": 71.3982, "district": "Saryarka", "ridership_24h": 2100},
    {"id": "S006", "name": "Mega Silk Way", "lat": 51.0891, "lon": 71.4050, "district": "Saryarka", "ridership_24h": 2800},
    {"id": "S007", "name": "Triathlon Park", "lat": 51.1200, "lon": 71.4500, "district": "Almaty", "ridership_24h": 950},
    {"id": "S008", "name": "Presidential Park", "lat": 51.1250, "lon": 71.4650, "district": "Almaty", "ridership_24h": 1200},
    {"id": "S009", "name": "Central Park", "lat": 51.1400, "lon": 71.4550, "district": "Almaty", "ridership_24h": 1750},
    {"id": "S010", "name": "Talan Towers", "lat": 51.1280, "lon": 71.4350, "district": "Esil", "ridership_24h": 2400},
    {"id": "S011", "name": "Expo 2017", "lat": 51.0895, "lon": 71.4170, "district": "Saryarka", "ridership_24h": 1650},
    {"id": "S012", "name": "Duman", "lat": 51.1450, "lon": 71.4200, "district": "Esil", "ridership_24h": 1100},
]

STATION_CAPACITY = 3000  # Estimated max capacity per station


def _get_stations(db: Session):
    """Shared helper: return stations list with DB fallback to mock data."""
    try:
        db_stations = db.query(StationORM).all()
        if db_stations:
            return [
                {"id": s.stop_id, "name": s.name, "lat": s.lat, "lon": s.lon,
                 "district": s.district, "ridership_24h": s.ridership_24h}
                for s in db_stations
            ]
    except Exception:
        pass
    # Return copies to avoid mutating mock data
    return [dict(s) for s in MOCK_STATIONS]


def _calc_load_pct(ridership_24h: int, hour: int) -> int:
    """Estimate load percentage for a given hour based on 24h ridership."""
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return min(95, int(ridership_24h / STATION_CAPACITY * 100 * 1.4))
    elif 6 <= hour <= 22:
        return min(70, int(ridership_24h / STATION_CAPACITY * 100 * 0.8))
    else:
        return min(30, int(ridership_24h / STATION_CAPACITY * 100 * 0.25))


@router.get("")
def list_stations(hour: Optional[int] = Query(None, ge=0, le=23), db: Session = Depends(get_db)):
    """List stations, optionally with heatmap load data for a specific hour."""
    stations = _get_stations(db)

    if hour is not None:
        for s in stations:
            ridership = s.get("ridership_24h", 0) or 0
            s["load_percent"] = _calc_load_pct(ridership, hour)

    return {"stations": stations}


@router.get("/{station_id}/forecast")
def get_station_forecast(station_id: str):
    forecast = get_forecast(station_id)
    return {"station_id": station_id, "forecast": forecast}


@router.get("/{station_id}/detail")
def get_station_detail(station_id: str, db: Session = Depends(get_db)):
    """Station detail with forecasts, connected routes, and active alerts."""
    from backend.models_orm import RouteORM, RouteStopORM, AlertORM
    from backend.services.alert_service import list_alerts

    try:
        station = db.query(StationORM).filter(StationORM.stop_id == station_id).first()
    except Exception:
        station = None

    station_info = None
    if station:
        station_info = {"id": station.stop_id, "name": station.name, "lat": station.lat,
                        "lon": station.lon, "district": station.district, "ridership_24h": station.ridership_24h}
    else:
        for s in MOCK_STATIONS:
            if s["id"] == station_id:
                station_info = dict(s)
                break
    if not station_info:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    # Connected routes
    connected_routes = []
    try:
        route_stops = db.query(RouteStopORM).filter(RouteStopORM.station_id == station_id).all()
        for rs in route_stops:
            route = db.query(RouteORM).filter(RouteORM.route_id == rs.route_id).first()
            if route:
                connected_routes.append({"id": route.route_id, "name": route.name, "color": route.color})
    except Exception:
        from backend.routers.routes import ROUTE_STOPS, MOCK_ROUTES
        for rid, stops in ROUTE_STOPS.items():
            if any(stop["id"] == station_id for stop in stops):
                route_info = next((r for r in MOCK_ROUTES if r["id"] == rid), None)
                if route_info:
                    connected_routes.append({"id": route_info["id"], "name": route_info["name"], "color": route_info.get("color", "#2E86AB")})

    # Forecast
    forecast = get_forecast(station_id)

    # Alerts for this station
    station_alerts = []
    try:
        alerts = db.query(AlertORM).filter(AlertORM.station_id == station_id).all()
        station_alerts = [{"severity": a.severity, "title": a.title, "message": a.message} for a in alerts]
    except Exception:
        pass

    # Hourly ridership pattern
    hourly = [{"hour": i, "ridership": f["predicted"]} for i, f in enumerate(forecast)]

    return {
        "station": station_info,
        "connected_routes": connected_routes,
        "forecast": forecast,
        "alerts": station_alerts,
        "hourly_ridership": hourly,
    }
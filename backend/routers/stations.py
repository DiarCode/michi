from typing import Optional
from fastapi import APIRouter, Depends, Query
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


@router.get("")
def list_stations(db: Session = Depends(get_db)):
    try:
        db_stations = db.query(StationORM).all()
        if db_stations:
            return {"stations": [
                {"id": s.stop_id, "name": s.name, "lat": s.lat, "lon": s.lon,
                 "district": s.district, "ridership_24h": s.ridership_24h}
                for s in db_stations
            ]}
    except Exception:
        pass
    return {"stations": MOCK_STATIONS}


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
        if not station:
            return {"error": "Station not found", "station_id": station_id}
    except Exception:
        station = None

    station_info = None
    if station:
        station_info = {"id": station.stop_id, "name": station.name, "lat": station.lat,
                        "lon": station.lon, "district": station.district, "ridership_24h": station.ridership_24h}
    else:
        for s in MOCK_STATIONS:
            if s["id"] == station_id:
                station_info = s
                break
    if not station_info:
        return {"error": "Station not found", "station_id": station_id}

    # Connected routes
    connected_routes = []
    try:
        route_stops = db.query(RouteStopORM).filter(RouteStopORM.station_id == station_id).all()
        for rs in route_stops:
            route = db.query(RouteORM).filter(RouteORM.route_id == rs.route_id).first()
            if route:
                connected_routes.append({"id": route.route_id, "name": route.name, "color": route.color})
    except Exception:
        for rid, stops in [
            ("R1", []), ("R2", []), ("R3", []), ("R4", []), ("R5", [])
        ]:
            from backend.routers.routes import ROUTE_STOPS
            for stop in ROUTE_STOPS.get(rid, []):
                if stop["id"] == station_id:
                    connected_routes.append({"id": rid, "name": f"Route {rid[1:]}", "color": "#2E86AB"})

    # Forecast
    forecast = get_forecast(station_id)

    # Alerts for this station
    station_alerts = []
    try:
        alerts = db.query(AlertORM).filter(AlertORM.station_id == station_id).all()
        station_alerts = [{"severity": a.severity, "title": a.title, "message": a.message} for a in alerts]
    except Exception:
        pass

    # Hourly ridership pattern (synthetic from forecast)
    hourly = [{"hour": i, "ridership": f["predicted"]} for i, f in enumerate(forecast)]

    return {
        "station": station_info,
        "connected_routes": connected_routes,
        "forecast": forecast,
        "alerts": station_alerts,
        "hourly_ridership": hourly,
    }


@router.get("")
def list_stations_with_heatmap(hour: Optional[int] = None, db: Session = Depends(get_db)):
    """List stations with heatmap data for a specific hour."""
    stations_data = list_stations(db)
    if isinstance(stations_data, dict) and "stations" in stations_data:
        stations = stations_data["stations"]
    else:
        return stations_data

    if hour is None:
        return stations_data

    # Add load percentage for heatmap
    for s in stations:
        ridership = s.get("ridership_24h", 0) or 0
        # Estimate hourly load: distribute 24h ridership using rush-hour curve
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            load_pct = min(95, int(ridership * 0.08 / 30))  # Rush hour
        elif 6 <= hour <= 22:
            load_pct = min(70, int(ridership * 0.04 / 30))  # Regular
        else:
            load_pct = min(30, int(ridership * 0.01 / 30))  # Night
        s["load_percent"] = load_pct

    return {"stations": stations, "hour": hour}

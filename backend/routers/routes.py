from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models_orm import RouteORM, RouteStopORM, StationORM
from backend.models import RouteListResponse, RouteStopsResponse, RouteForecastResponse, RouteScheduleResponse

router = APIRouter()

MOCK_ROUTES = [
    {"id": "R1", "name": "Route 12", "color": "#2E86AB", "stop_count": 4, "avg_ridership": 2100},
    {"id": "R2", "name": "Route 18", "color": "#A23B72", "stop_count": 4, "avg_ridership": 1850},
    {"id": "R3", "name": "Route 25", "color": "#F18F01", "stop_count": 4, "avg_ridership": 1600},
    {"id": "R4", "name": "Route 31", "color": "#C73E1D", "stop_count": 4, "avg_ridership": 1400},
    {"id": "R5", "name": "Route 40", "color": "#3B1F2B", "stop_count": 4, "avg_ridership": 1300},
]

ROUTE_STOPS = {
    "R1": [{"id": "S001", "name": "Nurly Zhol Station"}, {"id": "S003", "name": "Bayterek"}, {"id": "S010", "name": "Talan Towers"}],
    "R2": [{"id": "S002", "name": "Khan Shatyr"}, {"id": "S003", "name": "Bayterek"}, {"id": "S007", "name": "Triathlon Park"}],
    "R3": [{"id": "S004", "name": "Astana Arena"}, {"id": "S005", "name": "Nazarbayev University"}, {"id": "S006", "name": "Mega Silk Way"}],
    "R4": [{"id": "S001", "name": "Nurly Zhol Station"}, {"id": "S009", "name": "Central Park"}, {"id": "S008", "name": "Presidential Park"}],
    "R5": [{"id": "S012", "name": "Duman"}, {"id": "S010", "name": "Talan Towers"}, {"id": "S003", "name": "Bayterek"}],
}


@router.get("", response_model=RouteListResponse)
def list_routes(db: Session = Depends(get_db)):
    try:
        db_routes = db.query(RouteORM).all()
        if db_routes:
            return {"routes": [
                {"id": r.route_id, "name": r.name, "color": r.color,
                 "stop_count": r.stop_count, "avg_ridership": r.avg_ridership}
                for r in db_routes
            ]}
    except Exception:
        pass
    return {"routes": [dict(r) for r in MOCK_ROUTES]}


@router.get("/{route_id}/stops", response_model=RouteStopsResponse)
def get_route_stops(route_id: str, db: Session = Depends(get_db)):
    try:
        db_stops = (db.query(RouteStopORM).filter(RouteStopORM.route_id == route_id)
                    .order_by(RouteStopORM.stop_order).all())
        if db_stops:
            result = []
            for rs in db_stops:
                station = db.query(StationORM).filter(StationORM.stop_id == rs.station_id).first()
                result.append({"id": rs.station_id, "name": station.name if station else rs.station_id})
            return {"route_id": route_id, "stops": result}
    except Exception:
        pass
    return {"route_id": route_id, "stops": [dict(s) for s in ROUTE_STOPS.get(route_id, [])]}


@router.get("/{route_id}/forecast", response_model=RouteForecastResponse)
def get_route_forecast(route_id: str, db: Session = Depends(get_db)):
    """Aggregated route-level forecast averaging across all stops on the route."""
    from backend.services.forecast_service import get_forecast
    stops_data = get_route_stops(route_id, db)
    stops = stops_data.get("stops", [])

    if not stops:
        return {"route_id": route_id, "forecast": [], "avg_ridership": 0}

    all_forecasts = []
    for stop in stops:
        f = get_forecast(stop["id"])
        all_forecasts.append(f)

    hourly = []
    for h in range(24):
        preds = [f[h]["predicted"] for f in all_forecasts if len(f) > h]
        confs = [f[h]["confidence"] for f in all_forecasts if len(f) > h]
        if preds:
            hourly.append({
                "hour": h,
                "predicted": int(sum(preds) / len(preds)),
                "confidence": round(sum(confs) / len(confs), 3),
            })

    route_info = None
    try:
        r = db.query(RouteORM).filter(RouteORM.route_id == route_id).first()
        if r:
            route_info = {"id": r.route_id, "name": r.name, "color": r.color}
    except Exception:
        for r in MOCK_ROUTES:
            if r["id"] == route_id:
                route_info = r

    return {
        "route_id": route_id,
        "route": route_info,
        "stop_count": len(stops),
        "forecast": hourly,
        "avg_ridership": int(sum(h["predicted"] for h in hourly) / max(len(hourly), 1)),
    }


@router.get("/{route_id}/schedule", response_model=RouteScheduleResponse)
def get_route_schedule(route_id: str, db: Session = Depends(get_db)):
    """Generate a timetable for a route with departure times."""
    stops_data = get_route_stops(route_id, db)
    stops = stops_data.get("stops", [])
    if not stops:
        stops = ROUTE_STOPS.get(route_id, [])
    route_info = next((r for r in MOCK_ROUTES if r["id"] == route_id), None)
    route_name = route_info["name"] if route_info else route_id

    HEADWAY = 8
    FIRST_BUS = 6
    LAST_BUS = 23
    schedule = []
    for hour in range(FIRST_BUS, LAST_BUS + 1):
        for stop_idx, stop in enumerate(stops):
            offset_min = stop_idx * 3
            total_minutes = hour * 60 + offset_min
            t_hour = total_minutes // 60
            t_min = total_minutes % 60
            # Skip times past midnight
            if t_hour > 23:
                continue
            direction = "outbound" if t_hour < 14 else "inbound"
            schedule.append({
                "stop_id": stop["id"],
                "stop_name": stop["name"],
                "time": f"{t_hour:02d}:{t_min:02d}",
                "headway_min": HEADWAY,
                "direction": direction,
            })

    return {
        "route_id": route_id,
        "route_name": route_name,
        "stops": stops,
        "schedule": schedule,
        "first_bus": f"{FIRST_BUS:02d}:00",
        "last_bus": f"{LAST_BUS:02d}:30",
        "headway_min": HEADWAY,
    }
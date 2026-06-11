from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models import RouteForecastResponse, RouteListResponse, RouteScheduleResponse, RouteStopsResponse
from backend.models_orm import RouteORM, RouteStopORM, StationORM

router = APIRouter()


@router.get("", response_model=RouteListResponse)
def list_routes(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    """List all routes from the database."""
    db_routes = db.query(RouteORM).offset(offset).limit(limit).all()
    return {
        "routes": [
            {
                "id": r.route_id,
                "name": r.name,
                "color": r.color,
                "stop_count": r.stop_count,
                "avg_ridership": r.avg_ridership,
            }
            for r in db_routes
        ]
    }


@router.get("/{route_id}/stops", response_model=RouteStopsResponse)
def get_route_stops(route_id: str, db: Session = Depends(get_db_session)):
    """Get stops for a route from the database. Uses joinedload to avoid N+1 queries."""
    db_stops = db.query(RouteStopORM).filter(RouteStopORM.route_id == route_id).order_by(RouteStopORM.stop_order).all()
    # Batch-load all station names in a single query instead of N+1
    station_ids = [rs.station_id for rs in db_stops]
    station_map = {s.stop_id: s.name for s in db.query(StationORM).filter(StationORM.stop_id.in_(station_ids)).all()}
    result = [{"id": rs.station_id, "name": station_map.get(rs.station_id, rs.station_id)} for rs in db_stops]
    return {"route_id": route_id, "stops": result}


@router.get("/{route_id}/forecast", response_model=RouteForecastResponse)
def get_route_forecast(route_id: str, db: Session = Depends(get_db_session)):
    """Aggregated route-level forecast averaging across all stops on the route."""
    from backend.services.forecast_service import get_forecast

    stops_data = get_route_stops(route_id, db)
    stops = stops_data.get("stops", [])

    if not stops:
        return {"route_id": route_id, "forecast": [], "avg_ridership": 0}

    all_forecasts = []
    for stop in stops:
        f = get_forecast(stop["id"], db=db)
        all_forecasts.append(f)

    hourly = []
    for h in range(24):
        preds = [f[h]["predicted"] for f in all_forecasts if len(f) > h]
        confs = [f[h]["confidence"] for f in all_forecasts if len(f) > h]
        if preds:
            hourly.append(
                {
                    "hour": h,
                    "predicted": int(sum(preds) / len(preds)),
                    "confidence": round(sum(confs) / len(confs), 3),
                }
            )

    route_info = None
    r = db.query(RouteORM).filter(RouteORM.route_id == route_id).first()
    if r:
        route_info = {"id": r.route_id, "name": r.name, "color": r.color}

    return {
        "route_id": route_id,
        "route": route_info,
        "stop_count": len(stops),
        "forecast": hourly,
        "avg_ridership": int(sum(h["predicted"] for h in hourly) / max(len(hourly), 1)),
    }


@router.get("/{route_id}/schedule", response_model=RouteScheduleResponse)
def get_route_schedule(route_id: str, db: Session = Depends(get_db_session)):
    """Generate a timetable for a route with departure times."""
    stops_data = get_route_stops(route_id, db)
    stops = stops_data.get("stops", [])

    route = db.query(RouteORM).filter(RouteORM.route_id == route_id).first()
    route_name = route.name if route else route_id

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
            schedule.append(
                {
                    "stop_id": stop["id"],
                    "stop_name": stop["name"],
                    "time": f"{t_hour:02d}:{t_min:02d}",
                    "headway_min": HEADWAY,
                    "direction": direction,
                }
            )

    return {
        "route_id": route_id,
        "route_name": route_name,
        "stops": stops,
        "schedule": schedule,
        "first_bus": f"{FIRST_BUS:02d}:00",
        "last_bus": f"{LAST_BUS:02d}:30",
        "headway_min": HEADWAY,
    }

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models_orm import RouteORM, RouteStopORM, StationORM

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


@router.get("")
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
    return {"routes": MOCK_ROUTES}


@router.get("/{route_id}/stops")
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
    return {"route_id": route_id, "stops": ROUTE_STOPS.get(route_id, [])}

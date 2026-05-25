from fastapi import APIRouter

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
def list_routes():
    return {"routes": MOCK_ROUTES}

@router.get("/{route_id}/stops")
def get_route_stops(route_id: str):
    return {"route_id": route_id, "stops": ROUTE_STOPS.get(route_id, [])}

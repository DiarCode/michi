"""Seed database with sample Astana stations and routes."""
from sqlalchemy.orm import Session
from backend.database import Base, engine
from backend.models_orm import StationORM, RouteORM, RouteStopORM

ASTANA_STATIONS = [
    {"stop_id": "S001", "name": "Nurly Zhol Station", "lat": 51.1605, "lon": 71.4704, "district": "Esil", "ridership_24h": 1840},
    {"stop_id": "S002", "name": "Khan Shatyr", "lat": 51.1334, "lon": 71.4244, "district": "Esil", "ridership_24h": 3200},
    {"stop_id": "S003", "name": "Bayterek", "lat": 51.1283, "lon": 71.4305, "district": "Esil", "ridership_24h": 4100},
    {"stop_id": "S004", "name": "Astana Arena", "lat": 51.1081, "lon": 71.4024, "district": "Saryarka", "ridership_24h": 1500},
    {"stop_id": "S005", "name": "Nazarbayev University", "lat": 51.0906, "lon": 71.3982, "district": "Saryarka", "ridership_24h": 2100},
    {"stop_id": "S006", "name": "Mega Silk Way", "lat": 51.0891, "lon": 71.4050, "district": "Saryarka", "ridership_24h": 2800},
    {"stop_id": "S007", "name": "Triathlon Park", "lat": 51.1200, "lon": 71.4500, "district": "Almaty", "ridership_24h": 950},
    {"stop_id": "S008", "name": "Presidential Park", "lat": 51.1250, "lon": 71.4650, "district": "Almaty", "ridership_24h": 1200},
    {"stop_id": "S009", "name": "Central Park", "lat": 51.1400, "lon": 71.4550, "district": "Almaty", "ridership_24h": 1750},
    {"stop_id": "S010", "name": "Talan Towers", "lat": 51.1280, "lon": 71.4350, "district": "Esil", "ridership_24h": 2400},
    {"stop_id": "S011", "name": "Expo 2017", "lat": 51.0895, "lon": 71.4170, "district": "Saryarka", "ridership_24h": 1650},
    {"stop_id": "S012", "name": "Duman", "lat": 51.1450, "lon": 71.4200, "district": "Esil", "ridership_24h": 1100},
]

ASTANA_ROUTES = [
    {"route_id": "R1", "name": "Route 12", "color": "#2E86AB", "stop_count": 4, "avg_ridership": 2100},
    {"route_id": "R2", "name": "Route 18", "color": "#A23B72", "stop_count": 4, "avg_ridership": 1850},
    {"route_id": "R3", "name": "Route 25", "color": "#F18F01", "stop_count": 4, "avg_ridership": 1600},
    {"route_id": "R4", "name": "Route 31", "color": "#C73E1D", "stop_count": 4, "avg_ridership": 1400},
    {"route_id": "R5", "name": "Route 40", "color": "#3B1F2B", "stop_count": 4, "avg_ridership": 1300},
]

ROUTE_STOPS_DATA = [
    ("R1", "S001", 1), ("R1", "S003", 2), ("R1", "S010", 3), ("R1", "S012", 4),
    ("R2", "S002", 1), ("R2", "S003", 2), ("R2", "S007", 3), ("R2", "S008", 4),
    ("R3", "S004", 1), ("R3", "S005", 2), ("R3", "S006", 3), ("R3", "S011", 4),
    ("R4", "S001", 1), ("R4", "S009", 2), ("R4", "S008", 3), ("R4", "S007", 4),
    ("R5", "S012", 1), ("R5", "S010", 2), ("R5", "S003", 3), ("R5", "S002", 4),
]


def seed():
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        existing = session.query(StationORM).count()
        if existing > 0:
            print(f"Database already seeded with {existing} stations. Skipping.")
            return
        for s in ASTANA_STATIONS:
            session.add(StationORM(**s))
        for r in ASTANA_ROUTES:
            session.add(RouteORM(**r))
        for route_id, station_id, order in ROUTE_STOPS_DATA:
            session.add(RouteStopORM(route_id=route_id, station_id=station_id, stop_order=order))
        session.commit()
        print(f"Seeded {len(ASTANA_STATIONS)} stations, {len(ASTANA_ROUTES)} routes, {len(ROUTE_STOPS_DATA)} route stops.")


if __name__ == "__main__":
    seed()

"""Seed database with sample Astana stations and routes."""

from sqlalchemy.orm import Session

from backend.database import Base, engine
from backend.models import Station, Route


ASTANA_STATIONS = [
    {"id": "S001", "name": "Nurly Zhol Station", "lat": 51.1605, "lon": 71.4704, "district": "Esil"},
    {"id": "S002", "name": "Khan Shatyr", "lat": 51.1334, "lon": 71.4244, "district": "Esil"},
    {"id": "S003", "name": "Bayterek", "lat": 51.1283, "lon": 71.4305, "district": "Esil"},
    {"id": "S004", "name": "Astana Arena", "lat": 51.1081, "lon": 71.4024, "district": "Saryarka"},
    {"id": "S005", "name": "Nazarbayev University", "lat": 51.0906, "lon": 71.3982, "district": "Saryarka"},
    {"id": "S006", "name": "Mega Silk Way", "lat": 51.0891, "lon": 71.4050, "district": "Saryarka"},
    {"id": "S007", "name": "Triathlon Park", "lat": 51.1200, "lon": 71.4500, "district": "Almaty"},
    {"id": "S008", "name": "Presidential Park", "lat": 51.1250, "lon": 71.4650, "district": "Almaty"},
    {"id": "S009", "name": "Central Park", "lat": 51.1400, "lon": 71.4550, "district": "Almaty"},
    {"id": "S010", "name": "Talan Towers", "lat": 51.1280, "lon": 71.4350, "district": "Esil"},
    {"id": "S011", "name": "Expo 2017", "lat": 51.0895, "lon": 71.4170, "district": "Saryarka"},
    {"id": "S012", "name": "Duman", "lat": 51.1450, "lon": 71.4200, "district": "Esil"},
]

ASTANA_ROUTES = [
    {"id": "R1", "name": "Route 12", "color": "#2E86AB", "stop_ids": ["S001", "S003", "S010", "S012"]},
    {"id": "R2", "name": "Route 18", "color": "#A23B72", "stop_ids": ["S002", "S003", "S007", "S008"]},
    {"id": "R3", "name": "Route 25", "color": "#F18F01", "stop_ids": ["S004", "S005", "S006", "S011"]},
    {"id": "R4", "name": "Route 31", "color": "#C73E1D", "stop_ids": ["S001", "S009", "S008", "S007"]},
    {"id": "R5", "name": "Route 40", "color": "#3B1F2B", "stop_ids": ["S012", "S010", "S003", "S002"]},
]


def seed():
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        existing = session.query(Station).count()
        if existing > 0:
            print(f"Database already seeded with {existing} stations. Skipping.")
            return
        for s in ASTANA_STATIONS:
            session.add(Station(**s))
        for r in ASTANA_ROUTES:
            session.add(Route(**r))
        session.commit()
        print(f"Seeded {len(ASTANA_STATIONS)} stations and {len(ASTANA_ROUTES)} routes.")


if __name__ == "__main__":
    seed()

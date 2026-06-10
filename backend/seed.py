"""Seed database with real Astana transit data from OSM.

Uses pre-generated seed data from data/cache/astana_network_seed.json.
Falls back to a small hardcoded dataset if the file is missing.
"""
import json
import logging
from datetime import UTC
from pathlib import Path

logger = logging.getLogger(__name__)

from backend.database import SessionLocal
from backend.models_orm import RouteORM, RouteStopORM, StationORM

SEED_PATH = Path(__file__).parent / "data" / "cache" / "astana_network_seed.json"

FALLBACK_STATIONS = [
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

FALLBACK_ROUTES = [
    {"route_id": "R12", "name": "Route 12", "color": "#2E86AB", "stop_count": 4, "avg_ridership": 2100.0},
    {"route_id": "R18", "name": "Route 18", "color": "#A23B72", "stop_count": 4, "avg_ridership": 1850.0},
    {"route_id": "R25", "name": "Route 25", "color": "#F18F01", "stop_count": 4, "avg_ridership": 1600.0},
    {"route_id": "R31", "name": "Route 31", "color": "#C73E1D", "stop_count": 4, "avg_ridership": 1400.0},
    {"route_id": "R40", "name": "Route 40", "color": "#3B7A57", "stop_count": 4, "avg_ridership": 1300.0},
]

FALLBACK_ROUTE_STOPS = [
    ("R12", "S001", 1), ("R12", "S003", 2), ("R12", "S010", 3), ("R12", "S012", 4),
    ("R18", "S002", 1), ("R18", "S003", 2), ("R18", "S007", 3), ("R18", "S008", 4),
    ("R25", "S004", 1), ("R25", "S005", 2), ("R25", "S006", 3), ("R25", "S011", 4),
    ("R31", "S001", 1), ("R31", "S009", 2), ("R31", "S008", 3), ("R31", "S007", 4),
    ("R40", "S012", 1), ("R40", "S010", 2), ("R40", "S003", 3), ("R40", "S002", 4),
]


def load_seed_data():
    """Load seed data from OSM-generated file, falling back to hardcoded data."""
    if SEED_PATH.exists():
        with open(SEED_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded OSM data: %d stations, %d routes",
                    data['metadata']['total_stations'], data['metadata']['total_routes'])
        return data["stations"], data["routes"], data["route_stops"]

    # Try the old relative path as a secondary fallback
    OLD_SEED_PATH = Path(__file__).parent.parent / "data" / "cache" / "astana_network_seed.json"
    if OLD_SEED_PATH.exists():
        with open(OLD_SEED_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data["stations"], data["routes"], data["route_stops"]

    logger.warning("OSM seed not found at %s, using fallback data", SEED_PATH)
    return FALLBACK_STATIONS, FALLBACK_ROUTES, FALLBACK_ROUTE_STOPS


def seed():
    session = SessionLocal()
    from backend.models_orm import AlertORM, InterventionORM, RidershipORM
    try:
        existing = session.query(StationORM).count()
        # If we have very few stations (likely fallbacks), force a re-seed if the seed file is available
        stations, routes, route_stops = load_seed_data()

        if existing >= len(stations) and existing > 12:
            logger.info("Database already seeded with %d stations. Skipping.", existing)
            return

        if existing > 0:
            logger.info("Current station count (%d) is low. Clearing and re-seeding with full dataset.", existing)
            # Simple way to clear tables for a fresh seed
            session.query(RouteStopORM).delete()
            session.query(RouteORM).delete()
            session.query(StationORM).delete()
            session.commit()

        for s in stations:
            session.add(StationORM(**s))
        for r in routes:
            session.add(RouteORM(**r))
        for rs in route_stops:
            if isinstance(rs, dict):
                session.add(RouteStopORM(route_id=rs["route_id"], station_id=rs["station_id"], stop_order=rs["stop_order"]))
            else:
                route_id, station_id, order = rs
                session.add(RouteStopORM(route_id=route_id, station_id=station_id, stop_order=order))

        # Seed Executive Dashboard tables
        from datetime import datetime, timedelta
        now = datetime.now(UTC)

        # Ridership
        for i in range(30):
            session.add(RidershipORM(station_id="S001", timestamp=now - timedelta(days=i), passengers=1000 + i*10))

        # Interventions
        session.add(InterventionORM(intervention_type="re-route", created_at=now, status="completed"))
        session.add(InterventionORM(intervention_type="frequency", created_at=now, status="pending"))

        # Alerts
        session.add(AlertORM(severity="critical", title="Station Closed", created_at=now, station_id="S002"))

        session.commit()
        logger.info("Seeded %d stations, %d routes, %d route stops and dashboard data.",
                    len(stations), len(routes), len(route_stops))
    except Exception as e:
        session.rollback()
        logger.error("Seed failed: %s", e, exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    # In some environments, we might want to create all here, but usually Alembic handles it.
    # Base.metadata.create_all(bind=engine)
    seed()

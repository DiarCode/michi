"""Shared test fixtures for Michi backend tests."""
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app import app
from backend.database import Base

TEST_DB_URL = "sqlite:///./test_michi.db"


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def db(engine):
    """Function-scoped DB session with auto-rollback for test isolation."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")()
    # Nest a SAVEPOINT so we can rollback after each test
    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="session")
def client(engine):
    import backend.database as db_mod
    from backend.database import SessionLocal as _SL
    from backend.database import engine as _eng

    SessionLocal = sessionmaker(bind=engine)
    db_mod.engine = engine
    db_mod.SessionLocal = SessionLocal

    c = TestClient(app)
    yield c

    db_mod.engine = _eng
    db_mod.SessionLocal = _SL


@pytest.fixture
def seed_stations(db):
    from backend.models_orm import StationORM
    stations = []
    for i in range(5):
        s = StationORM(stop_id=f"TST{i:03d}", name=f"Test Station {i}",
                       lat=51.1 + i * 0.01, lon=71.4 + i * 0.01,
                       district=f"District {i}", ridership_24h=1000 + i * 200)
        db.add(s)
        stations.append(s)
    db.flush()
    return stations


@pytest.fixture
def seed_routes(db, seed_stations):
    from backend.models_orm import RouteORM, RouteStopORM
    r1 = RouteORM(route_id="TR01", name="Test Route 1", color="#FF0000", stop_count=3, avg_ridership=500)
    r2 = RouteORM(route_id="TR02", name="Test Route 2", color="#0000FF", stop_count=2, avg_ridership=300)
    db.add_all([r1, r2])
    db.flush()
    stops = [
        RouteStopORM(route_id="TR01", station_id="TST000", stop_order=0),
        RouteStopORM(route_id="TR01", station_id="TST001", stop_order=1),
        RouteStopORM(route_id="TR01", station_id="TST002", stop_order=2),
        RouteStopORM(route_id="TR02", station_id="TST001", stop_order=0),
        RouteStopORM(route_id="TR02", station_id="TST003", stop_order=1),
    ]
    db.add_all(stops)
    db.flush()
    return [r1, r2]


@pytest.fixture
def seed_alerts(db, seed_stations):
    from backend.models_orm import AlertORM
    alerts = [
        AlertORM(severity="critical", title="Overcrowding at TST000",
                  message="Platform capacity exceeded",
                  station_id="TST000", family="crowding", what="Platform overcrowding",
                  when_hint="Peak hours", where_hint="TST000",
                  why="Ridership exceeds capacity", confidence=0.92,
                  consequence_if_ignored="Passenger safety risk",
                  sla_timer_minutes=15, created_at=datetime.now(UTC)),
        AlertORM(severity="high", title="Delay on Test Route 1",
                  message="Bus delayed by 15 min",
                  route_id="TR01", family="delay", what="Service delay",
                  when_hint="Current", where_hint="Route TR01",
                  why="Traffic congestion", confidence=0.85,
                  consequence_if_ignored="Passenger wait times increase",
                  sla_timer_minutes=30, created_at=datetime.now(UTC)),
        AlertORM(severity="medium", title="Weather advisory",
                  message="Snow expected", family="weather",
                  what="Severe weather", when_hint="Tonight",
                  where_hint="All routes", why="Blizzard forecast",
                  confidence=0.78, consequence_if_ignored="Service disruption",
                  sla_timer_minutes=60, created_at=datetime.now(UTC)),
    ]
    db.add_all(alerts)
    db.flush()
    return alerts


@pytest.fixture
def seed_forecasts(db, seed_stations):
    from backend.models_orm import ForecastORM
    forecasts = []
    now = datetime.now(UTC)
    for s in seed_stations:
        for h in [15, 30, 60, 120]:
            forecasts.append(ForecastORM(
                station_id=s.stop_id,
                timestamp=now + timedelta(minutes=h),
                predicted=float(50 + hash(s.stop_id) % 100),
                confidence=0.85 - h / 600.0,
                model_version="dts-gssf",
                created_at=now,
                horizon_minutes=h,
            ))
    db.add_all(forecasts)
    db.flush()
    return forecasts

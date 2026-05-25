"""Shared test fixtures — SQLite in-memory DB for all backend tests."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base
from backend.models_orm import StationORM, RouteORM, RouteStopORM, AlertORM


@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def seeded_session(db_session):
    """Pre-populated DB with 3 stations and 2 routes."""
    for s in [
        StationORM(stop_id="S001", name="Test Station A", lat=51.16, lon=71.47, district="Esil", ridership_24h=1000),
        StationORM(stop_id="S002", name="Test Station B", lat=51.13, lon=71.42, district="Almaty", ridership_24h=2000),
        StationORM(stop_id="S003", name="Test Station C", lat=51.09, lon=71.40, district="Saryarka", ridership_24h=500),
    ]:
        db_session.add(s)
    for r in [
        RouteORM(route_id="R1", name="Route 1", color="#FF0000", stop_count=2, avg_ridership=1500.0),
        RouteORM(route_id="R2", name="Route 2", color="#00FF00", stop_count=2, avg_ridership=800.0),
    ]:
        db_session.add(r)
    db_session.add(RouteStopORM(route_id="R1", station_id="S001", stop_order=1))
    db_session.add(RouteStopORM(route_id="R1", station_id="S002", stop_order=2))
    db_session.add(RouteStopORM(route_id="R2", station_id="S002", stop_order=1))
    db_session.add(RouteStopORM(route_id="R2", station_id="S003", stop_order=2))
    db_session.commit()
    return db_session

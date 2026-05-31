"""Database configuration — defaults to local SQLite for development."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default to local SQLite file; override with DATABASE_URL for production (e.g. PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./michi.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db_session():
    """FastAPI dependency that provides a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Backward-compatible alias
get_db = get_db_session


def init_db():
    """Seed initial data. Alembic handles all table creation/migration."""
    # Import all ORM models so they register with Base.metadata
    from backend.models_orm import (  # noqa: F401
        StationORM, RouteORM, RouteStopORM, AlertORM, RidershipORM, ForecastORM,
        HistoricalRidershipORM, WeatherReadingORM, EventORM, InterventionORM,
        ModelArtifactORM, PredictionAccuracyORM,
    )
    from backend.seed import seed
    seed()

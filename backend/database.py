"""Database configuration — defaults to local SQLite for development."""

import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    echo=False,
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite to allow concurrent reads from multiple processes
# (backend, celery worker, celery beat). WAL prevents "database is locked" errors.
if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


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
        AlertORM,
        EventORM,
        ForecastORM,
        HistoricalRidershipORM,
        InterventionORM,
        ModelArtifactORM,
        PredictionAccuracyORM,
        RidershipORM,
        RouteORM,
        RouteStopORM,
        StationORM,
        WeatherReadingORM,
    )
    from backend.seed import seed

    seed()

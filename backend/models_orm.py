"""SQLAlchemy ORM models for Michi database."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from backend.database import Base


class StationORM(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True, index=True)
    stop_id = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    district = Column(String(100))
    ridership_24h = Column(Integer, default=0)


class RouteORM(Base):
    __tablename__ = "routes"
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    color = Column(String(10))
    stop_count = Column(Integer, default=0)
    avg_ridership = Column(Float, default=0.0)


class RouteStopORM(Base):
    __tablename__ = "route_stops"
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(String(20), ForeignKey("routes.route_id"), nullable=False)
    station_id = Column(String(20), ForeignKey("stations.stop_id"), nullable=False)
    stop_order = Column(Integer, nullable=False)


class AlertORM(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    severity = Column(String(20), nullable=False)
    title = Column(String(300), nullable=False)
    message = Column(Text)
    station_id = Column(String(20), ForeignKey("stations.stop_id"))
    route_id = Column(String(20), ForeignKey("routes.route_id"))
    created_at = Column(DateTime, nullable=False)


class RidershipORM(Base):
    __tablename__ = "ridership"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(20), ForeignKey("stations.stop_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    passengers = Column(Integer, nullable=False)


class ForecastORM(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(20), ForeignKey("stations.stop_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    predicted = Column(Float, nullable=False)
    confidence = Column(Float, default=0.0)
    model_version = Column(String(50))
    created_at = Column(DateTime)

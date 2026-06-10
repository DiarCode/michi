"""SQLAlchemy ORM models for Michi database."""
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text

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
    # Rich alert fields
    family = Column(String(50))
    what = Column(Text)
    when_hint = Column(String(200))
    where_hint = Column(String(200))
    why = Column(Text)
    confidence = Column(Float, default=0.0)
    consequence_if_ignored = Column(Text)
    sla_timer_minutes = Column(Integer, default=30)
    acknowledged = Column(Boolean, default=False)
    assigned_to = Column(String(100))


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
    horizon_minutes = Column(Integer, default=60)
    route_id = Column(String(20))


class HistoricalRidershipORM(Base):
    __tablename__ = "historical_ridership"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(20), ForeignKey("stations.stop_id"), nullable=False, index=True)
    route_id = Column(String(20), ForeignKey("routes.route_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    passengers_boarding = Column(Integer, nullable=False)
    passengers_alighting = Column(Integer, nullable=False)
    load = Column(Integer, nullable=False)
    weather_code = Column(String(10))
    temperature = Column(Float)
    is_holiday = Column(Boolean, default=False)
    is_event_day = Column(Boolean, default=False)
    day_of_week = Column(Integer)
    hour = Column(Integer)


class WeatherReadingORM(Base):
    __tablename__ = "weather_readings"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    temperature = Column(Float)
    precipitation = Column(Float)
    wind_speed = Column(Float)
    visibility = Column(Float)
    weather_code = Column(String(10))
    sudden_change = Column(Boolean, default=False)


class EventORM(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(300), nullable=False)
    venue = Column(String(200))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    expected_attendance = Column(Integer)
    affected_routes = Column(Text)  # JSON array
    affected_stations = Column(Text)  # JSON array
    event_type = Column(String(50))


class InterventionORM(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    intervention_type = Column(String(50), nullable=False)
    route_id = Column(String(20), ForeignKey("routes.route_id"))
    station_id = Column(String(20), ForeignKey("stations.stop_id"))
    created_at = Column(DateTime, nullable=False)
    status = Column(String(20), default="pending")  # pending, approved, executing, completed, cancelled
    operator_note = Column(Text)
    predicted_impact = Column(Text)  # JSON
    actual_impact = Column(Text)  # JSON
    approved_by = Column(String(100))


class ModelArtifactORM(Base):
    __tablename__ = "model_artifacts"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)
    artifact_path = Column(String(500), nullable=False)
    metrics_json = Column(Text)  # JSON: {mae, rmse, mape, mase}
    training_config_json = Column(Text)  # JSON
    dataset_hash = Column(String(64))
    feature_version = Column(Integer, default=1)
    created_at = Column(DateTime, nullable=False)
    is_production = Column(Boolean, default=False)
    is_shadow = Column(Boolean, default=False)


class PredictionAccuracyORM(Base):
    __tablename__ = "prediction_accuracy"
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), ForeignKey("model_artifacts.version"))
    station_id = Column(String(20), ForeignKey("stations.stop_id"))
    route_id = Column(String(20), ForeignKey("routes.route_id"))
    forecast_timestamp = Column(DateTime, nullable=False)
    horizon_minutes = Column(Integer, nullable=False)
    predicted = Column(Float, nullable=False)
    actual = Column(Float)
    absolute_error = Column(Float)
    mape = Column(Float)
    evaluated_at = Column(DateTime)

"""Pydantic response models aligned with ORM schema."""

from typing import Any

from pydantic import BaseModel


class StationResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    district: str | None = None
    ridership_24h: int | None = None
    load_percent: int | None = None


class StationListResponse(BaseModel):
    stations: list[StationResponse]
    hour: int | None = None


class StationDetailResponse(BaseModel):
    station: StationResponse
    connected_routes: list[dict[str, Any]]
    forecast: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    hourly_ridership: list[dict[str, Any]]


class RouteResponse(BaseModel):
    id: str
    name: str
    color: str | None = None
    stop_count: int | None = None
    avg_ridership: float | None = None


class RouteListResponse(BaseModel):
    routes: list[RouteResponse]


class RouteStopResponse(BaseModel):
    id: str
    name: str


class RouteStopsResponse(BaseModel):
    route_id: str
    stops: list[RouteStopResponse]


class RouteForecastResponse(BaseModel):
    route_id: str
    route: RouteResponse | None = None
    stop_count: int
    forecast: list[dict[str, Any]]
    avg_ridership: int


class ScheduleEntry(BaseModel):
    stop_id: str
    stop_name: str
    time: str
    headway_min: int
    direction: str


class RouteScheduleResponse(BaseModel):
    route_id: str
    route_name: str
    stops: list[RouteStopResponse]
    schedule: list[ScheduleEntry]
    first_bus: str
    last_bus: str
    headway_min: int


class ForecastPoint(BaseModel):
    station_id: str
    timestamp: str
    predicted: int
    confidence: float


class ForecastResponse(BaseModel):
    station_id: str
    forecast: list[ForecastPoint]


class AlertResponse(BaseModel):
    id: int
    severity: str
    title: str
    message: str
    station_id: str | None = None
    route_id: str | None = None
    created_at: str
    acknowledged: bool = False
    auto: bool | None = None
    rule_id: str | None = None


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]


class KPIResponse(BaseModel):
    total_stations: int
    active_routes: int
    avg_ridership: float
    alerts_today: int
    on_time_performance: float | None = None
    peak_hour: str | None = None


class OperationsReportResponse(BaseModel):
    date: str
    kpis: KPIResponse
    district_summary: dict[str, dict[str, int]]
    peak_hours: list[str]
    over_capacity_stations: list[dict[str, Any]]
    total_stations: int


class ScenarioDeltaPoint(BaseModel):
    station_id: str
    station_name: str
    baseline: float
    perturbed: float
    delta: float
    delta_pct: float


class ScenarioSummary(BaseModel):
    total_ridership_change: float
    most_affected_station: str
    least_affected_station: str


class ScenarioResult(BaseModel):
    scenario_id: str
    baseline_forecasts: list[dict[str, Any]]
    perturbed_forecasts: list[dict[str, Any]]
    deltas: list[ScenarioDeltaPoint]
    summary: ScenarioSummary


class WeatherReadingResponse(BaseModel):
    id: int | None = None
    timestamp: str | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    precipitation_mm: float | None = None
    weather_code: int | None = None
    description: str | None = None
    is_forecast: bool = False
    source: str = "open-meteo"


class WeatherImpactResponse(BaseModel):
    weather_code: int
    temperature_c: float | None = None
    description: str = ""
    impact_factor: float = 1.0


class ReportRequest(BaseModel):
    format: str = "pdf"  # "pdf" or "csv"
    period: int = 30  # days: 7, 30, 90

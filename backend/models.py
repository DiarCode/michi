"""Pydantic response models aligned with ORM schema."""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class StationResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    district: Optional[str] = None
    ridership_24h: Optional[int] = None
    load_percent: Optional[int] = None


class StationListResponse(BaseModel):
    stations: List[StationResponse]
    hour: Optional[int] = None


class StationDetailResponse(BaseModel):
    station: StationResponse
    connected_routes: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    hourly_ridership: List[Dict[str, Any]]


class RouteResponse(BaseModel):
    id: str
    name: str
    color: Optional[str] = None
    stop_count: Optional[int] = None
    avg_ridership: Optional[float] = None


class RouteListResponse(BaseModel):
    routes: List[RouteResponse]


class RouteStopResponse(BaseModel):
    id: str
    name: str


class RouteStopsResponse(BaseModel):
    route_id: str
    stops: List[RouteStopResponse]


class RouteForecastResponse(BaseModel):
    route_id: str
    route: Optional[RouteResponse] = None
    stop_count: int
    forecast: List[Dict[str, Any]]
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
    stops: List[RouteStopResponse]
    schedule: List[ScheduleEntry]
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
    forecast: List[ForecastPoint]


class AlertResponse(BaseModel):
    id: int
    severity: str
    title: str
    message: str
    station_id: Optional[str] = None
    route_id: Optional[str] = None
    created_at: str
    acknowledged: bool = False
    auto: Optional[bool] = None
    rule_id: Optional[str] = None


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]


class KPIResponse(BaseModel):
    total_stations: int
    active_routes: int
    avg_ridership: float
    alerts_today: int
    on_time_performance: Optional[float] = None
    peak_hour: Optional[str] = None


class OperationsReportResponse(BaseModel):
    date: str
    kpis: KPIResponse
    district_summary: Dict[str, Dict[str, int]]
    peak_hours: List[str]
    over_capacity_stations: List[Dict[str, Any]]
    total_stations: int


class ScenarioResult(BaseModel):
    scenario_id: str
    base_metrics: Dict[str, float]
    scenario_metrics: Dict[str, float]
    changes: Dict[str, float]
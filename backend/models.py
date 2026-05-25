from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Station(BaseModel):
    id: int
    stop_id: str
    name: str
    lat: float
    lon: float
    district: Optional[str] = None
    route_ids: List[int] = []

class Route(BaseModel):
    id: int
    route_id: str
    name: str
    color: Optional[str] = None
    stop_sequence: List[int] = []

class Alert(BaseModel):
    id: int
    alert_type: str
    severity: str
    station_id: Optional[int] = None
    route_id: Optional[int] = None
    message: str
    acknowledged: bool = False
    created_at: datetime

class KPIDashboard(BaseModel):
    total_stations: int
    active_routes: int
    avg_ridership: float
    alerts_today: int

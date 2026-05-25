from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models_orm import StationORM
from backend.services.forecast_service import get_forecast

router = APIRouter()

MOCK_STATIONS = [
    {"id": "S001", "name": "Nurly Zhol Station", "lat": 51.1605, "lon": 71.4704, "district": "Esil", "ridership_24h": 1840},
    {"id": "S002", "name": "Khan Shatyr", "lat": 51.1334, "lon": 71.4244, "district": "Esil", "ridership_24h": 3200},
    {"id": "S003", "name": "Bayterek", "lat": 51.1283, "lon": 71.4305, "district": "Esil", "ridership_24h": 4100},
    {"id": "S004", "name": "Astana Arena", "lat": 51.1081, "lon": 71.4024, "district": "Saryarka", "ridership_24h": 1500},
    {"id": "S005", "name": "Nazarbayev University", "lat": 51.0906, "lon": 71.3982, "district": "Saryarka", "ridership_24h": 2100},
    {"id": "S006", "name": "Mega Silk Way", "lat": 51.0891, "lon": 71.4050, "district": "Saryarka", "ridership_24h": 2800},
    {"id": "S007", "name": "Triathlon Park", "lat": 51.1200, "lon": 71.4500, "district": "Almaty", "ridership_24h": 950},
    {"id": "S008", "name": "Presidential Park", "lat": 51.1250, "lon": 71.4650, "district": "Almaty", "ridership_24h": 1200},
    {"id": "S009", "name": "Central Park", "lat": 51.1400, "lon": 71.4550, "district": "Almaty", "ridership_24h": 1750},
    {"id": "S010", "name": "Talan Towers", "lat": 51.1280, "lon": 71.4350, "district": "Esil", "ridership_24h": 2400},
    {"id": "S011", "name": "Expo 2017", "lat": 51.0895, "lon": 71.4170, "district": "Saryarka", "ridership_24h": 1650},
    {"id": "S012", "name": "Duman", "lat": 51.1450, "lon": 71.4200, "district": "Esil", "ridership_24h": 1100},
]


@router.get("")
def list_stations(db: Session = Depends(get_db)):
    try:
        db_stations = db.query(StationORM).all()
        if db_stations:
            return {"stations": [
                {"id": s.stop_id, "name": s.name, "lat": s.lat, "lon": s.lon,
                 "district": s.district, "ridership_24h": s.ridership_24h}
                for s in db_stations
            ]}
    except Exception:
        pass
    return {"stations": MOCK_STATIONS}


@router.get("/{station_id}/forecast")
def get_station_forecast(station_id: str):
    forecast = get_forecast(station_id)
    return {"station_id": station_id, "forecast": forecast}

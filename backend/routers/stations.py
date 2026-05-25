from fastapi import APIRouter

router = APIRouter()

@router.get("")
def list_stations():
    return {"stations": []}

@router.get("/{station_id}/forecast")
def get_station_forecast(station_id: str):
    return {"station_id": station_id, "forecast": []}

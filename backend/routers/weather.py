"""Weather API router — exposes Open-Meteo weather data to the dashboard."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.weather_service import (
    fetch_forecast_weather,
    get_latest_weather,
    get_weather_history,
    get_weather_impact_factor,
)

router = APIRouter()


@router.get("/current")
def get_current_weather(db: Session = Depends(get_db)):
    """Return the latest weather reading for Astana.

    If cached data is less than 30 minutes old, returns it directly.
    Otherwise fetches fresh data from Open-Meteo.
    """
    return get_latest_weather(db)


@router.get("/forecast")
def get_forecast(hours: int = Query(default=24, ge=1, le=72), db: Session = Depends(get_db)):
    """Return hourly weather forecast for Astana."""
    forecast = fetch_forecast_weather(db, hours=hours)
    return {"forecast": forecast, "hours": hours}


@router.get("/impact")
def get_weather_impact(db: Session = Depends(get_db)):
    """Return the current weather impact factor for ridership.

    The factor is a multiplier (0.8-1.3) representing how weather
    affects transit ridership. Extreme weather increases ridership.
    """
    weather = get_latest_weather(db)
    code = weather.get("weather_code", 0) or 0
    temp = weather.get("temperature_c") or 20.0
    factor = get_weather_impact_factor(code, temp)
    return {
        "weather_code": code,
        "temperature_c": temp,
        "description": weather.get("description", ""),
        "impact_factor": factor,
    }


@router.get("/history")
def get_weather_history_endpoint(hours: int = Query(default=24, ge=1, le=168), db: Session = Depends(get_db)):
    """Return recent weather history for Astana."""
    history = get_weather_history(db, hours=hours)
    return {"history": history, "hours": hours}

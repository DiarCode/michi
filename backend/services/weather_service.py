"""Weather data service — fetches from Open-Meteo API and stores in DB."""

import logging
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy.orm import Session

from backend.models_orm import WeatherReadingORM

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ASTANA_LAT = 51.1694
ASTANA_LON = 71.4491

# WMO weather code mapping
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _describe(code: int) -> str:
    """Return human-readable description for a WMO weather code."""
    return WMO_CODES.get(code, f"Unknown ({code})")


def fetch_current_weather(db: Session) -> dict:
    """Fetch current weather for Astana from Open-Meteo and store in DB.

    Returns the current reading as a dict. Falls back to cached data on failure.
    """
    params = {
        "latitude": ASTANA_LAT,
        "longitude": ASTANA_LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "timezone": "Asia/Almaty",
    }
    try:
        resp = httpx.get(OPEN_METEO_URL, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        code = int(current.get("weather_code", 0))
        reading = WeatherReadingORM(
            timestamp=datetime.now(UTC),
            temperature_c=current.get("temperature_2m"),
            humidity_pct=current.get("relative_humidity_2m"),
            wind_speed_kmh=current.get("wind_speed_10m"),
            precipitation_mm=0.0,
            weather_code=code,
            description=_describe(code),
            is_forecast=False,
            source="open-meteo",
        )
        db.add(reading)
        db.commit()
        db.refresh(reading)
        return _orm_to_dict(reading)
    except Exception as e:
        logger.warning("Open-Meteo current weather fetch failed: %s — returning cached", e)
        return _get_latest_cached(db, is_forecast=False)


def fetch_forecast_weather(db: Session, hours: int = 24) -> list[dict]:
    """Fetch hourly forecast for Astana from Open-Meteo and store in DB.

    Returns the forecast list as dicts. Falls back to cached data on failure.
    """
    params = {
        "latitude": ASTANA_LAT,
        "longitude": ASTANA_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
        "forecast_hours": hours,
        "timezone": "Asia/Almaty",
    }
    try:
        resp = httpx.get(OPEN_METEO_URL, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        precip = hourly.get("precipitation", [])
        codes = hourly.get("weather_code", [])

        results = []
        for i in range(min(len(times), hours)):
            code = int(codes[i]) if i < len(codes) else 0
            reading = WeatherReadingORM(
                timestamp=datetime.fromisoformat(times[i]).replace(tzinfo=UTC) if times else datetime.now(UTC),
                temperature_c=temps[i] if i < len(temps) else None,
                humidity_pct=humidities[i] if i < len(humidities) else None,
                wind_speed_kmh=winds[i] if i < len(winds) else None,
                precipitation_mm=precip[i] if i < len(precip) else 0.0,
                weather_code=code,
                description=_describe(code),
                is_forecast=True,
                source="open-meteo",
            )
            db.add(reading)
            results.append(_orm_to_dict(reading))

        db.commit()
        return results
    except Exception as e:
        logger.warning("Open-Meteo forecast fetch failed: %s — returning cached", e)
        return _get_latest_cached_list(db, is_forecast=True, hours=hours)


def get_weather_impact_factor(weather_code: int, temperature_c: float | None) -> float:
    """Return a ridership impact multiplier (0.8–1.3) based on weather.

    Extreme weather (heavy snow, thunderstorm) increases transit ridership.
    Clear/sunny → baseline (1.0).
    """
    # Temperature extremes push people toward transit
    if temperature_c is not None:
        if temperature_c < -20:
            temp_factor = 1.25  # Very cold: more transit
        elif temperature_c < -10:
            temp_factor = 1.15
        elif temperature_c > 35:
            temp_factor = 1.1  # Very hot: more transit (AC)
        elif temperature_c > 30:
            temp_factor = 1.05
        else:
            temp_factor = 1.0
    else:
        temp_factor = 1.0

    # Weather code factor: extreme weather → higher ridership on transit
    if weather_code in (95, 96, 99):  # Thunderstorm
        code_factor = 1.3
    elif weather_code in (65, 75, 82):  # Heavy rain / heavy snow / violent showers
        code_factor = 1.25
    elif weather_code in (63, 73, 81):  # Moderate rain/snow
        code_factor = 1.15
    elif weather_code in (61, 71, 80) or weather_code in (45, 48):  # Slight rain/snow/showers
        code_factor = 1.1
    elif weather_code in (51, 53, 55):  # Drizzle
        code_factor = 1.05
    elif weather_code in (0, 1) or weather_code in (2,):  # Clear / mainly clear
        code_factor = 1.0
    elif weather_code in (3,):  # Overcast
        code_factor = 1.05
    else:
        code_factor = 1.0

    # Combine: take the stronger effect
    return round(max(temp_factor, code_factor), 2)


def get_latest_weather(db: Session) -> dict:
    """Get the most recent weather reading from DB.

    Returns cached data if less than 30 minutes old, otherwise fetches fresh.
    """
    latest = (
        db.query(WeatherReadingORM)
        .filter(WeatherReadingORM.is_forecast == False)  # noqa: E712
        .order_by(WeatherReadingORM.timestamp.desc())
        .first()
    )
    if latest and latest.timestamp:
        age = datetime.now(UTC) - (
            latest.timestamp.replace(tzinfo=UTC) if latest.timestamp.tzinfo is None else latest.timestamp
        )
        if age < timedelta(minutes=30):
            return _orm_to_dict(latest)
    # Stale or missing — fetch fresh
    return fetch_current_weather(db)


def get_weather_history(db: Session, hours: int = 24) -> list[dict]:
    """Get weather history from DB for the past N hours."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows = (
        db.query(WeatherReadingORM)
        .filter(WeatherReadingORM.timestamp >= cutoff)
        .order_by(WeatherReadingORM.timestamp.asc())
        .all()
    )
    return [_orm_to_dict(r) for r in rows]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _orm_to_dict(reading: WeatherReadingORM) -> dict:
    return {
        "id": reading.id,
        "timestamp": reading.timestamp.isoformat() if reading.timestamp else None,
        "temperature_c": reading.temperature_c,
        "humidity_pct": reading.humidity_pct,
        "wind_speed_kmh": reading.wind_speed_kmh,
        "precipitation_mm": reading.precipitation_mm,
        "weather_code": reading.weather_code,
        "description": reading.description,
        "is_forecast": reading.is_forecast,
        "source": reading.source,
    }


def _get_latest_cached(db: Session, is_forecast: bool = False) -> dict:
    """Return the latest cached reading as a fallback dict."""
    latest = (
        db.query(WeatherReadingORM)
        .filter(WeatherReadingORM.is_forecast == is_forecast)
        .order_by(WeatherReadingORM.timestamp.desc())
        .first()
    )
    if latest:
        return _orm_to_dict(latest)
    return {
        "id": None,
        "timestamp": datetime.now(UTC).isoformat(),
        "temperature_c": None,
        "humidity_pct": None,
        "wind_speed_kmh": None,
        "precipitation_mm": 0.0,
        "weather_code": 0,
        "description": "No data available",
        "is_forecast": is_forecast,
        "source": "cache-miss",
    }


def _get_latest_cached_list(db: Session, is_forecast: bool, hours: int) -> list[dict]:
    """Return cached forecast list as a fallback."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows = (
        db.query(WeatherReadingORM)
        .filter(WeatherReadingORM.is_forecast == is_forecast)
        .filter(WeatherReadingORM.timestamp >= cutoff)
        .order_by(WeatherReadingORM.timestamp.asc())
        .all()
    )
    if rows:
        return [_orm_to_dict(r) for r in rows]
    return []

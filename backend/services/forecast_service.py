"""Forecast service - generates and retrieves ridership forecasts."""
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List

MOCK_FORECAST: Dict[str, List[dict]] = {}


def generate_24h_forecast(station_id: str, base_ridership: int = 1000) -> List[dict]:
    """Generate a 24-hour forecast for a station."""
    now = datetime.now(timezone.utc)
    hourly = []
    for h in range(24):
        ts = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if h < now.hour:
            ts += timedelta(days=1)
        factor = 0.3 + 0.7 * max(0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 22 else 0.1
        predicted = int(base_ridership * factor + np.random.randint(-50, 50))
        confidence = round(0.85 + np.random.random() * 0.12, 3)
        hourly.append({
            "station_id": station_id,
            "timestamp": ts.isoformat(),
            "predicted": max(0, predicted),
            "confidence": confidence,
        })
    return hourly


def get_forecast(station_id: str) -> List[dict]:
    if station_id not in MOCK_FORECAST:
        MOCK_FORECAST[station_id] = generate_24h_forecast(station_id)
    return MOCK_FORECAST[station_id]


def get_kpi_metrics() -> dict:
    return {
        "total_stations": 12,
        "active_routes": 5,
        "avg_ridership": 1845.0,
        "alerts_today": 2,
        "on_time_performance": 94.2,
        "peak_hour": "08:00",
    }

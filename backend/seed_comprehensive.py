"""Comprehensive seed script for Michi transit platform.

Seeds rich historical data so the dashboard, timeline, forecasts, and simulation
all work with real-looking data from day one. Run on fresh DB or re-seed to refresh.

Usage:
    python -m backend.seed_comprehensive
    # Or via docker compose:
    docker compose exec backend python -m backend.seed_comprehensive
"""

import json
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SEED_PATH = Path(__file__).parent / "data" / "cache" / "astana_network_seed.json"

# Kazakh public holidays (month, day)
KAZAKH_HOLIDAYS = frozenset({
    (1, 1), (1, 2), (1, 7), (3, 8), (3, 22), (3, 23),
    (5, 1), (5, 7), (5, 9), (6, 10), (7, 6), (8, 30),
    (10, 25), (12, 16), (12, 17),
})

# Astana monthly average temperature (°C) and precipitation (mm)
ASTANA_MONTHLY_CLIMATE = {
    1:  (-14.2, 15), 2:  (-13.5, 12), 3:  (-5.0, 13),  4:  (7.5, 18),
    5:  (15.0, 22),  6:  (21.0, 25),  7:  (23.5, 30),  8:  (21.5, 18),
    9:  (14.0, 14),  10: (5.0, 20),   11: (-4.5, 18),  12: (-11.0, 16),
}

# WMO weather codes weighted by season for Astana
WMO_CODES_SUMMER = [0, 0, 0, 1, 1, 2, 2, 3, 3, 51, 61]
WMO_CODES_WINTER = [0, 0, 1, 2, 3, 3, 71, 71, 75, 75, 51]
WMO_CODES_SPRING = [0, 0, 1, 1, 2, 3, 3, 51, 61, 71, 80]

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain",
    65: "Heavy rain", 71: "Slight snow", 73: "Moderate snow",
    75: "Heavy snow", 80: "Rain showers", 82: "Violent rain showers",
    95: "Thunderstorm",
}

random.seed(42)
np.random.seed(42)


def _is_holiday(dt: datetime) -> bool:
    """Check if a date is a Kazakh public holiday or weekend."""
    if dt.weekday() >= 5:
        return True
    return (dt.month, dt.day) in KAZAKH_HOLIDAYS


def _rush_factor(hour: int, is_weekend: bool) -> float:
    """Compute rush-hour ridership multiplier for a given hour."""
    if is_weekend:
        if 10 <= hour <= 14:
            return 0.6
        elif 8 <= hour <= 20:
            return 0.4
        else:
            return 0.15
    # Weekday
    if 7 <= hour <= 8:
        return 2.5
    elif hour == 9:
        return 2.0
    elif 17 <= hour <= 18:
        return 2.3
    elif hour == 19:
        return 1.8
    elif 12 <= hour <= 13:
        return 1.2
    elif 6 <= hour <= 20:
        return 0.7
    else:
        return 0.15


def _seasonal_factor(month: int) -> float:
    """Seasonal ridership multiplier (winter higher due to less walking)."""
    if month in [11, 12, 1, 2]:
        return 1.15
    elif month in [6, 7, 8]:
        return 0.85
    return 1.0


def _generate_temperature(month: int, hour: int) -> float:
    """Generate realistic Astana temperature for a given month/hour."""
    avg_temp, _ = ASTANA_MONTHLY_CLIMATE.get(month, (10, 15))
    # Diurnal variation: coldest around 5am, warmest around 3pm
    diurnal = 8 * math.sin(2 * math.pi * (hour - 5) / 24)
    noise = np.random.normal(0, 2)
    return round(avg_temp + diurnal + noise, 1)


def _generate_weather_code(month: int) -> int:
    """Generate a realistic WMO weather code for a given month."""
    if month in [6, 7, 8]:
        codes = WMO_CODES_SUMMER
    elif month in [11, 12, 1, 2]:
        codes = WMO_CODES_WINTER
    else:
        codes = WMO_CODES_SPRING
    return random.choice(codes)


def _generate_precipitation(code: int, month: int) -> float:
    """Generate precipitation in mm based on weather code and month."""
    _, avg_precip = ASTANA_MONTHLY_CLIMATE.get(month, (10, 15))
    if code in (61, 63, 65, 80, 82):
        return round(np.random.exponential(avg_precip / 10), 1)
    elif code in (71, 73, 75):
        return round(np.random.exponential(avg_precip / 8), 1)
    elif code in (51, 53, 55):
        return round(np.random.exponential(0.5), 1)
    return 0.0


def seed_comprehensive(days_back: int = 7, force: bool = False):
    """Seed comprehensive historical data for the dashboard.

    Args:
        days_back: Number of days of historical data to generate (default 7).
                   Use 30 for a full month or 90 for a quarter.
        force: If True, re-seed even if data already exists.
    """
    import sys
    from backend.database import SessionLocal
    from backend.models_orm import (
        AlertORM,
        ForecastORM,
        HistoricalRidershipORM,
        InterventionORM,
        ModelArtifactORM,
        PredictionAccuracyORM,
        RidershipORM,
        RouteORM,
        RouteStopORM,
        StationORM,
        WeatherReadingORM,
    )

    session = SessionLocal()
    try:
        # Idempotency guard: skip if data already exists
        existing_hist = session.query(HistoricalRidershipORM).count()
        if existing_hist > 0 and not force:
            logger.info(
                "Historical data already exists (%d records). Use --force to re-seed.",
                existing_hist,
            )
            return

        stations = session.query(StationORM).all()
        routes = session.query(RouteORM).all()
        route_stops = session.query(RouteStopORM).all()

        if not stations:
            logger.error("No stations found. Run basic seed first.")
            return

        n_stations = len(stations)
        logger.info("Found %d stations, %d routes for comprehensive seeding", n_stations, len(routes))

        # Build route→stations lookup for realistic route assignments
        route_station_map = {}
        for rs in route_stops:
            route_station_map.setdefault(rs.route_id, []).append(rs.station_id)

        # Assign each station a default route (first route that serves it)
        station_route = {}
        for route_id, sids in route_station_map.items():
            for sid in sids:
                if sid not in station_route:
                    station_route[sid] = route_id

        # If a station has no route, assign the first route
        default_route = routes[0].route_id if routes else "R12"
        for s in stations:
            if s.stop_id not in station_route:
                station_route[s.stop_id] = default_route

        now = datetime.now(UTC)
        start_time = now - timedelta(days=days_back)

        # ---------------------------------------------------------------
        # 1. Historical Ridership (hourly for each station × days_back)
        # ---------------------------------------------------------------
        existing_hist = session.query(HistoricalRidershipORM).count()
        if existing_hist > 0:
            logger.info("Clearing %d existing historical ridership records", existing_hist)
            session.query(HistoricalRidershipORM).delete()
            session.commit()

        total_hours = days_back * 24
        logger.info("Generating %d hours × %d stations of historical ridership...", total_hours, n_stations)

        hist_batch = []
        for day_offset in range(days_back):
            day = start_time + timedelta(days=day_offset)
            is_hol = _is_holiday(day)
            dow = day.weekday()

            for hour in range(24):
                ts = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                month = day.month

                for s in stations:
                    base_hourly = (s.ridership_24h or 1500) / 24.0
                    rush = _rush_factor(hour, is_hol)
                    seasonal = _seasonal_factor(month)
                    noise = 1.0 + np.random.normal(0, 0.08)
                    riders = max(1, int(base_hourly * rush * seasonal * noise))

                    # Boarding ≈ 60% of load, alighting ≈ 40%
                    boarding = max(1, int(riders * 0.6))
                    alighting = max(1, int(riders * 0.4))
                    load = riders

                    route_id = station_route.get(s.stop_id, default_route)

                    hist_batch.append(HistoricalRidershipORM(
                        station_id=s.stop_id,
                        route_id=route_id,
                        timestamp=ts,
                        passengers_boarding=boarding,
                        passengers_alighting=alighting,
                        load=load,
                        weather_code=str(_generate_weather_code(month)),
                        temperature=_generate_temperature(month, hour),
                        is_holiday=is_hol,
                        is_event_day=False,
                        day_of_week=dow,
                        hour=hour,
                    ))

                # Commit every 3 hours to avoid memory issues
                if hour % 3 == 0 and hist_batch:
                    session.bulk_save_objects(hist_batch)
                    session.commit()
                    hist_batch = []

        if hist_batch:
            session.bulk_save_objects(hist_batch)
            session.commit()

        hist_count = session.query(HistoricalRidershipORM).count()
        logger.info("Seeded %d historical ridership records", hist_count)

        # ---------------------------------------------------------------
        # 2. Weather Readings (hourly for days_back)
        # ---------------------------------------------------------------
        existing_weather = session.query(WeatherReadingORM).count()
        if existing_weather > 0:
            logger.info("Clearing %d existing weather records", existing_weather)
            session.query(WeatherReadingORM).delete()
            session.commit()

        weather_batch = []
        for day_offset in range(days_back):
            day = start_time + timedelta(days=day_offset)
            month = day.month
            for hour in range(24):
                ts = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                temp = _generate_temperature(month, hour)
                code = _generate_weather_code(month)
                precip = _generate_precipitation(code, month)
                humidity = round(np.random.uniform(30, 80), 1)
                wind = round(np.random.uniform(2, 25), 1)

                weather_batch.append(WeatherReadingORM(
                    timestamp=ts,
                    temperature_c=temp,
                    humidity_pct=humidity,
                    wind_speed_kmh=wind,
                    precipitation_mm=precip,
                    weather_code=code,
                    description=WMO_DESCRIPTIONS.get(code, "Unknown"),
                    is_forecast=False,
                    source="seed",
                ))

        session.bulk_save_objects(weather_batch)
        session.commit()
        logger.info("Seeded %d weather records", len(weather_batch))

        # ---------------------------------------------------------------
        # 3. Forecasts (4 horizons × every 3h × days_back for sample stations)
        # ---------------------------------------------------------------
        existing_forecasts = session.query(ForecastORM).count()
        if existing_forecasts > 0:
            logger.info("Clearing %d existing forecast records", existing_forecasts)
            session.query(ForecastORM).delete()
            session.commit()

        horizons = [15, 30, 60, 120]
        # Generate forecasts for every 3 hours for all stations
        forecast_batch = []
        for day_offset in range(days_back + 2):  # +2 for future forecasts
            day = start_time + timedelta(days=day_offset)
            is_hol = _is_holiday(day)
            month = day.month

            for hour in range(0, 24, 3):
                ts = day.replace(hour=hour, minute=0, second=0, microsecond=0)

                for s in stations:
                    base_hourly = (s.ridership_24h or 1500) / 24.0
                    rush = _rush_factor(hour, is_hol)
                    seasonal = _seasonal_factor(month)

                    for h_min in horizons:
                        h_hours = h_min / 60.0
                        target_hour = (hour + int(h_hours)) % 24
                        target_rush = _rush_factor(target_hour, is_hol)
                        pred = max(1, int(base_hourly * ((rush + target_rush) / 2) * seasonal * (1 + np.random.normal(0, 0.05))))
                        conf = max(0.5, min(0.98, 0.95 - h_hours * 0.02 + np.random.normal(0, 0.02)))

                        forecast_batch.append(ForecastORM(
                            station_id=s.stop_id,
                            timestamp=ts + timedelta(minutes=h_min),
                            predicted=float(pred),
                            confidence=round(conf, 3),
                            model_version="dts-gssf",
                            created_at=ts,
                            horizon_minutes=h_min,
                            route_id=station_route.get(s.stop_id, default_route),
                        ))

                # Commit every 6 hours to avoid memory issues
                if hour % 6 == 0 and forecast_batch:
                    session.bulk_save_objects(forecast_batch)
                    session.commit()
                    forecast_batch = []

        if forecast_batch:
            session.bulk_save_objects(forecast_batch)
            session.commit()

        fc_count = session.query(ForecastORM).count()
        logger.info("Seeded %d forecast records", fc_count)

        # ---------------------------------------------------------------
        # 4. Prediction Accuracy (daily for days_back)
        # ---------------------------------------------------------------
        existing_pa = session.query(PredictionAccuracyORM).count()
        if existing_pa > 0:
            logger.info("Clearing %d existing prediction accuracy records", existing_pa)
            session.query(PredictionAccuracyORM).delete()
            session.commit()

        pa_batch = []
        # Sample 20 stations for accuracy tracking to keep volume reasonable
        sample_stations = [s.stop_id for s in stations[:20]]
        for day_offset in range(days_back):
            day = start_time + timedelta(days=day_offset)
            for hour in range(0, 24, 3):
                ts = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                for sid in sample_stations:
                    for h_min in horizons:
                        actual = int(np.random.uniform(50, 500))
                        error = abs(np.random.normal(0, actual * 0.08))
                        pred = max(1, actual + int(np.random.normal(0, actual * 0.05)))

                        pa_batch.append(PredictionAccuracyORM(
                            model_version="dts-gssf",
                            station_id=sid,
                            route_id=station_route.get(sid, default_route),
                            forecast_timestamp=ts,
                            horizon_minutes=h_min,
                            predicted=float(pred),
                            actual=float(actual),
                            absolute_error=round(error, 2),
                            mape=round(error / max(actual, 1) * 100, 2),
                            evaluated_at=ts + timedelta(minutes=h_min),
                        ))

        session.bulk_save_objects(pa_batch)
        session.commit()
        logger.info("Seeded %d prediction accuracy records", len(pa_batch))

        # ---------------------------------------------------------------
        # 5. Model Artifact (register production model)
        # ---------------------------------------------------------------
        existing_artifact = session.query(ModelArtifactORM).filter(
            ModelArtifactORM.is_production == True
        ).first()
        if not existing_artifact:
            session.add(ModelArtifactORM(
                version="dts-gssf-v1",
                artifact_path="models/dts_gssf_v1.pt",
                metrics_json=json.dumps({"mae": 6.38, "rmse": 9.76, "mape": 4.2}),
                training_config_json=json.dumps({
                    "F_in": 16, "n_series": 400, "n_agg": 26,
                    "d_model": 192, "horizon": 4, "K": 3, "lora_r": 16,
                    "dropout": 0.1, "n_heads": 6,
                }),
                created_at=now - timedelta(days=30),
                is_production=True,
                is_shadow=False,
            ))
            session.commit()
            logger.info("Created production model artifact")

        # ---------------------------------------------------------------
        # 6. Rich Alerts (variety of realistic alerts)
        # ---------------------------------------------------------------
        existing_alerts = session.query(AlertORM).count()
        if existing_alerts < 10:
            alert_templates = [
                {"severity": "critical", "title": "Station overload detected", "family": "crowding",
                 "what": "Passenger count exceeds platform capacity", "why": "Rush hour surge + reduced frequency",
                 "where_hint": "Platform area", "confidence": 0.92},
                {"severity": "warning", "title": "Headway deviation on Route 12", "family": "schedule",
                 "what": "Bus intervals exceeding 20-minute threshold", "why": "Traffic congestion on Bayterek corridor",
                 "where_hint": "Route 12 corridor", "confidence": 0.85},
                {"severity": "warning", "title": "Bus delay approaching Bayterek", "family": "delay",
                 "what": "Vehicle 10+ minutes behind schedule", "why": "Road construction blocking lane",
                 "where_hint": "Bayterek Ave", "confidence": 0.78},
                {"severity": "info", "title": "Weather advisory: Snow expected", "family": "weather",
                 "what": "Heavy snow forecast for next 3 hours", "why": "Open-Meteo forecast",
                 "where_hint": "Citywide", "confidence": 0.90},
                {"severity": "critical", "title": "Vehicle breakdown on Route 42", "family": "maintenance",
                 "what": "Bus engine failure, passengers need transfer", "why": "Engine overheating, age of vehicle",
                 "where_hint": "Route 42, stop S015", "confidence": 0.95},
                {"severity": "warning", "title": "Ridership anomaly at Khan Shatyr", "family": "anomaly",
                 "what": "Ridership 40% above expected for this time", "why": "Shopping event nearby",
                 "where_hint": "Khan Shatyr Mall", "confidence": 0.82},
                {"severity": "info", "title": "Route 54 frequency adjustment", "family": "schedule",
                 "what": "Frequency reduced from 8min to 12min headway", "why": "Low demand period",
                 "where_hint": "Route 54", "confidence": 0.88},
                {"severity": "warning", "title": "Crowding at Nurly Zhol terminal", "family": "crowding",
                 "what": "Platform at 85% capacity during peak", "why": "Transfer hub concentration",
                 "where_hint": "Nurly Zhol Station", "confidence": 0.91},
                {"severity": "info", "title": "Weekend schedule change", "family": "schedule",
                 "what": "Modified weekend timetable active", "why": "Reduced weekend demand pattern",
                 "where_hint": "All routes", "confidence": 0.99},
                {"severity": "critical", "title": "Signal failure at intersection", "family": "infrastructure",
                 "what": "Traffic signal malfunction affecting Route 30", "why": "Power outage at intersection",
                 "where_hint": "Kabanbay Batyr / Mangilik El", "confidence": 0.87},
            ]

            for i, at in enumerate(alert_templates):
                station_id = stations[i % n_stations].stop_id
                route_id = station_route.get(station_id, default_route)
                session.add(AlertORM(
                    severity=at["severity"],
                    title=at["title"],
                    message=at.get("what", at["title"]),
                    station_id=station_id,
                    route_id=route_id,
                    created_at=now - timedelta(minutes=random.randint(1, 3600)),
                    family=at.get("family"),
                    what=at.get("what"),
                    when_hint=f"{random.randint(1, 59)}m ago",
                    where_hint=at.get("where_hint"),
                    why=at.get("why"),
                    confidence=at.get("confidence", 0.8),
                    acknowledged=i > 6,  # First 7 open, last 3 acknowledged
                ))
            session.commit()
            logger.info("Seeded %d rich alerts", len(alert_templates))

        # ---------------------------------------------------------------
        # 7. Update RidershipORM (executive dashboard needs 30 days)
        # ---------------------------------------------------------------
        existing_ridership = session.query(RidershipORM).count()
        if existing_ridership < 100:
            session.query(RidershipORM).delete()
            for day_offset in range(30):
                day = now - timedelta(days=day_offset)
                for s in stations[:20]:  # Top 20 stations for ridership records
                    base = s.ridership_24h or 1500
                    daily = int(base * _seasonal_factor(day.month) * (0.8 + np.random.normal(0, 0.05)))
                    session.add(RidershipORM(
                        station_id=s.stop_id,
                        timestamp=day,
                        passengers=daily,
                    ))
            session.commit()
            logger.info("Seeded ridership records for 30 days × 20 stations")

        session.commit()
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE SEED COMPLETE")
        logger.info("  Historical ridership: %d records", session.query(HistoricalRidershipORM).count())
        logger.info("  Weather readings:     %d records", session.query(WeatherReadingORM).count())
        logger.info("  Forecasts:            %d records", session.query(ForecastORM).count())
        logger.info("  Prediction accuracy:  %d records", session.query(PredictionAccuracyORM).count())
        logger.info("  Alerts:               %d records", session.query(AlertORM).count())
        logger.info("  Model artifacts:      %d records", session.query(ModelArtifactORM).count())
        logger.info("=" * 60)

    except Exception as e:
        session.rollback()
        logger.error("Comprehensive seed failed: %s", e, exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Seed comprehensive historical data")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data")
    parser.add_argument("--force", action="store_true", help="Re-seed even if data exists")
    args = parser.parse_args()
    seed_comprehensive(days_back=args.days, force=args.force)
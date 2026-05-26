"""Generate realistic historical ridership, weather, and event data for DTS-GSSF training.

Produces ~1.75M rows of hourly ridership data across 374 stations x 365 days x 24 hours,
plus weather readings and event calendar, and populates the database.

Usage:
    uv run python data/generate_historical.py
"""
import json
import math
import random
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEED_PATH = Path(__file__).parent / "cache" / "astana_network_seed.json"

# Astana seasonal patterns
SEASONAL_FACTORS = {
    1: 1.15, 2: 1.15, 3: 1.10,
    4: 1.0, 5: 0.95, 6: 0.90,
    7: 0.85, 8: 0.85, 9: 0.95,
    10: 1.05, 11: 1.10, 12: 1.15,
}

HOUR_FACTORS = {
    0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.10,
    6: 0.40, 7: 0.85, 8: 1.00, 9: 0.75, 10: 0.55, 11: 0.50,
    12: 0.60, 13: 0.55, 14: 0.50, 15: 0.55, 16: 0.70,
    17: 0.95, 18: 1.00, 19: 0.80, 20: 0.55, 21: 0.35,
    22: 0.20, 23: 0.10,
}

WEEKDAY_FACTOR = 1.0
SATURDAY_FACTOR = 0.7
SUNDAY_FACTOR = 0.5

# Weather templates: (code, temp_range, precip_range, wind_range, demand_modifier)
WEATHER_TEMPLATES = [
    ("clear", (-15, 35), (0, 0), (0, 15), 1.0),
    ("cloudy", (-10, 30), (0, 0), (5, 20), 1.02),
    ("rain", (5, 25), (1, 15), (10, 30), 1.10),
    ("snow", (-25, 5), (1, 20), (10, 35), 1.20),
    ("blizzard", (-30, -5), (5, 30), (30, 60), 1.25),
    ("fog", (-10, 20), (0, 0), (0, 10), 1.08),
    ("extreme_cold", (-40, -20), (0, 5), (15, 40), 1.25),
    ("heatwave", (35, 45), (0, 0), (5, 15), 0.92),
]

# Astana events
EVENT_TEMPLATES = [
    {"name": "Astana Arena Football Match", "venue": "Astana Arena", "type": "sports",
     "attendance": 25000, "routes": ["R12", "R25"], "stations": ["S004"], "duration_hours": 4},
    {"name": "Khan Shatyr Concert", "venue": "Khan Shatyr", "type": "concert",
     "attendance": 8000, "routes": ["R18", "R40"], "stations": ["S002"], "duration_hours": 3},
    {"name": "Expo Exhibition", "venue": "Expo 2017 Pavilion", "type": "exhibition",
     "attendance": 15000, "routes": ["R25", "R50"], "stations": ["S011"], "duration_hours": 8},
    {"name": "Independence Day Parade", "venue": "Nurzhol Blvd", "type": "ceremony",
     "attendance": 50000, "routes": ["R12", "R31", "R40"], "stations": ["S001", "S003"], "duration_hours": 5},
    {"name": "Nauryz Festival", "venue": "Central Park", "type": "festival",
     "attendance": 30000, "routes": ["R31", "R7"], "stations": ["S009"], "duration_hours": 10},
    {"name": "University Graduation", "venue": "Nazarbayev University", "type": "ceremony",
     "attendance": 5000, "routes": ["R25"], "stations": ["S005"], "duration_hours": 4},
    {"name": "Capital City Day", "venue": "Multiple venues", "type": "festival",
     "attendance": 60000, "routes": ["R12", "R18", "R31"], "stations": ["S002", "S003", "S010"], "duration_hours": 12},
    {"name": "Barys Hockey Game", "venue": "Barys Arena", "type": "sports",
     "attendance": 8000, "routes": ["R33", "R22"], "stations": [], "duration_hours": 3},
]

KAZAKH_HOLIDAYS = [
    (1, 1), (1, 2), (1, 7), (3, 8), (3, 22), (3, 23), (5, 1), (5, 7),
    (5, 9), (6, 10), (7, 6), (8, 30), (10, 25), (12, 16), (12, 17),
]


def load_stations_and_routes():
    if SEED_PATH.exists():
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["stations"], data["routes"], data["route_stops"]
    raise FileNotFoundError(f"Seed data not found at {SEED_PATH}")


def is_holiday(dt):
    return (dt.month, dt.day) in KAZAKH_HOLIDAYS


def get_weather_for_date(dt):
    month = dt.month
    if month in (12, 1, 2):
        weights = [0.15, 0.15, 0.05, 0.20, 0.10, 0.15, 0.15, 0.05]
    elif month in (6, 7, 8):
        weights = [0.45, 0.20, 0.15, 0.00, 0.00, 0.00, 0.00, 0.20]
    else:
        weights = [0.30, 0.25, 0.15, 0.05, 0.00, 0.05, 0.10, 0.10]
    idx = np.random.choice(len(WEATHER_TEMPLATES), p=weights)
    code, (t_lo, t_hi), (p_lo, p_hi), (w_lo, w_hi), demand_mod = WEATHER_TEMPLATES[idx]
    temp = round(random.uniform(t_lo, t_hi), 1)
    precip = round(random.uniform(p_lo, p_hi), 2) if p_hi > 0 else 0.0
    wind = round(random.uniform(w_lo, w_hi), 1)
    vis = max(0.5, 10.0 - wind * 0.15 - precip * 0.3)
    sudden = random.random() < 0.05
    return {
        "weather_code": code,
        "temperature": temp,
        "precipitation": precip,
        "wind_speed": wind,
        "visibility": round(vis, 1),
        "sudden_change": sudden,
        "demand_modifier": demand_mod,
    }


def generate_ridership_for_hour(station, route_id, dt, weather, is_event_nearby, station_order, total_stops):
    base = station.get("ridership_24h", 1500) / 24.0
    hour_f = HOUR_FACTORS.get(dt.hour, 0.3)
    month_f = SEASONAL_FACTORS.get(dt.month, 1.0)
    dow = dt.weekday()
    day_f = SUNDAY_FACTOR if dow == 6 else SATURDAY_FACTOR if dow == 5 else WEEKDAY_FACTOR
    if is_holiday(dt):
        day_f = SUNDAY_FACTOR
    weather_f = weather.get("demand_modifier", 1.0)
    event_f = 2.5 if is_event_nearby else 1.0
    order_ratio = (station_order + 1) / max(total_stops, 1)
    boarding_f = 1.3 - 0.6 * order_ratio if dt.hour in (7, 8, 17, 18) else 1.0
    alighting_f = 0.4 + 0.6 * order_ratio if dt.hour in (8, 9, 18, 19) else 0.5
    boarding = int(base * hour_f * month_f * day_f * weather_f * event_f * boarding_f * random.uniform(0.85, 1.15))
    alighting = int(base * hour_f * month_f * day_f * weather_f * event_f * alighting_f * random.uniform(0.80, 1.20))
    load = max(0, boarding - alighting + random.randint(-5, 10))
    if random.random() < 0.02:
        boarding = int(boarding * random.uniform(2.0, 4.0))
        load = int(load * random.uniform(1.5, 3.0))
    return max(0, boarding), max(0, alighting), max(0, load)


def generate_events(year):
    events = []
    for template in EVENT_TEMPLATES:
        count = random.randint(2, 6) if template["type"] in ("sports", "concert") else random.randint(1, 3)
        for _ in range(count):
            month = random.choice([3, 4, 5, 6, 7, 8, 9, 10, 12])
            day = random.randint(1, 28)
            try:
                start = datetime(year, month, day, random.randint(15, 19), 0, tzinfo=timezone.utc)
            except ValueError:
                continue
            end = start + timedelta(hours=template["duration_hours"])
            events.append({
                "name": template["name"],
                "venue": template["venue"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "expected_attendance": template["attendance"] + random.randint(-2000, 2000),
                "affected_routes": json.dumps(template["routes"]),
                "affected_stations": json.dumps(template.get("stations", [])),
                "event_type": template["type"],
            })
    return events


def build_route_station_map(route_stops):
    mapping = {}
    for route_id, station_id, order in route_stops:
        mapping.setdefault(route_id, []).append((station_id, order))
    for rid in mapping:
        mapping[rid].sort(key=lambda x: x[1])
    return mapping


def generate_all_data(days=365):
    random.seed(42)
    np.random.seed(42)
    stations, routes, route_stops = load_stations_and_routes()
    rs_map = build_route_station_map(route_stops)
    year = 2025
    start_date = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    events = generate_events(year)
    event_station_ids = set()
    for ev in events:
        event_station_ids.update(json.loads(ev["affected_stations"]))
    ridership_rows = []
    weather_rows = []
    current = start_date
    end = start_date + timedelta(days=days)
    hour_count = 0
    while current < end:
        weather = get_weather_for_date(current)
        weather_rows.append({
            "timestamp": current.isoformat(),
            "temperature": weather["temperature"],
            "precipitation": weather["precipitation"],
            "wind_speed": weather["wind_speed"],
            "visibility": weather["visibility"],
            "weather_code": weather["weather_code"],
            "sudden_change": weather["sudden_change"],
        })
        active_events = [e for e in events
                         if datetime.fromisoformat(e["start_time"]) - timedelta(hours=2) <= current <=
                         datetime.fromisoformat(e["end_time"]) + timedelta(hours=1)]
        active_stations = set()
        for ev in active_events:
            active_stations.update(json.loads(ev["affected_stations"]))
        for station in stations:
            sid = station["stop_id"]
            is_event_nearby = sid in active_stations
            for route_id, r_stops in rs_map.items():
                stop_idx = next((i for i, (s, _) in enumerate(r_stops) if s == sid), None)
                if stop_idx is None:
                    continue
                total_stops = len(r_stops)
                boarding, alighting, load = generate_ridership_for_hour(
                    station, route_id, current, weather, is_event_nearby, stop_idx, total_stops
                )
                ridership_rows.append({
                    "station_id": sid,
                    "route_id": route_id,
                    "timestamp": current.isoformat(),
                    "passengers_boarding": boarding,
                    "passengers_alighting": alighting,
                    "load": load,
                    "weather_code": weather["weather_code"],
                    "temperature": weather["temperature"],
                    "is_holiday": is_holiday(current),
                    "is_event_day": is_event_nearby,
                    "day_of_week": current.weekday(),
                    "hour": current.hour,
                })
        current += timedelta(hours=1)
        hour_count += 1
        if hour_count % 720 == 0:
            print(f"  Generated {hour_count // 24} days ({len(ridership_rows)} rows)...")
    print(f"Done: {len(ridership_rows)} ridership, {len(weather_rows)} weather, {len(events)} events")
    return ridership_rows, weather_rows, events


def populate_database(ridership_rows, weather_rows, events_data):
    from backend.database import SessionLocal, engine
    from backend.models_orm import HistoricalRidershipORM, WeatherReadingORM, EventORM
    from backend.database import Base
    from datetime import datetime

    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    # Convert ISO strings to datetime objects for SQLite compatibility
    for row in weather_rows:
        if isinstance(row.get("timestamp"), str):
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
    for row in ridership_rows:
        if isinstance(row.get("timestamp"), str):
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
    for ev in events_data:
        if isinstance(ev.get("start_time"), str):
            ev["start_time"] = datetime.fromisoformat(ev["start_time"])
        if isinstance(ev.get("end_time"), str):
            ev["end_time"] = datetime.fromisoformat(ev["end_time"])

    session = SessionLocal()
    try:
        existing = session.query(HistoricalRidershipORM).count()
        if existing > 0:
            print(f"Historical data exists ({existing} rows). Skipping.")
            return

        batch_size = 500
        print(f"Inserting {len(weather_rows)} weather readings...")
        for i in range(0, len(weather_rows), batch_size):
            for row in weather_rows[i:i + batch_size]:
                session.add(WeatherReadingORM(**row))
            session.commit()

        print(f"Inserting {len(events_data)} events...")
        for ev in events_data:
            session.add(EventORM(**ev))
        session.commit()

        print(f"Inserting {len(ridership_rows)} ridership rows...")
        for i in range(0, len(ridership_rows), batch_size):
            for row in ridership_rows[i:i + batch_size]:
                session.add(HistoricalRidershipORM(**row))
            if i % 10000 == 0 and i > 0:
                session.commit()
                print(f"  ...{i} rows committed")
        session.commit()
        print(f"Populated {len(ridership_rows)} ridership rows.")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("Generating historical data for DTS-GSSF training...")
    ridership, weather, events = generate_all_data(days=365)
    populate_database(ridership, weather, events)
    print("Done!")
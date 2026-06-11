"""Data loader — builds feature tensors from database for DTS-GSSF training/prediction.

Feature layout (F=16) matches training pipeline in main.py:
  0: passengers_boarding (lag-1)
  1: passengers_alighting
  2: load
  3: temperature
  4: precipitation
  5: is_holiday
  6: rush_hour
  7: delta_h (hours to next holiday)
  8: hour_sin
  9: hour_cos
 10: dow_sin
 11: dow_cos
 12: roll_6h (6-hour rolling mean of load)
 13: roll_24h (24-hour rolling mean of load)
 14: dev_24h (load - roll_24h)
 15: ratio_24h (load / roll_24h)
"""

from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from backend.models_orm import (
    EventORM,
    HistoricalRidershipORM,
    RouteStopORM,
    StationORM,
    WeatherReadingORM,
)

# Kazakh public holidays (month, day)
KAZAKH_HOLIDAYS = frozenset({
    (1, 1), (1, 2), (1, 7), (3, 8), (3, 22), (3, 23),
    (5, 1), (5, 7), (5, 9), (6, 10), (7, 6), (8, 30),
    (10, 25), (12, 16), (12, 17),
})


def _next_holiday_hours(ts: datetime) -> float:
    """Compute hours to the next Kazakh holiday from a given timestamp."""
    year = ts.year
    for y in range(year, year + 2):
        for (month, day) in KAZAKH_HOLIDAYS:
            h = datetime(y, month, day, tzinfo=UTC)
            if h >= ts:
                delta = (h - ts).total_seconds() / 3600.0
                return max(0.0, delta)
    return 720.0  # fallback: 30 days


def build_adjacency(session: Session) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Build physical adjacency matrix from route-station topology."""
    stations = session.query(StationORM).all()
    route_stops = session.query(RouteStopORM).all()
    station_idx = {s.stop_id: i for i, s in enumerate(stations)}
    N = len(stations)
    A = np.eye(N, dtype=np.float32)
    for rs in route_stops:
        route_stops_for_route = (
            session.query(RouteStopORM)
            .filter(RouteStopORM.route_id == rs.route_id)
            .order_by(RouteStopORM.stop_order)
            .all()
        )
        for i in range(len(route_stops_for_route) - 1):
            s1 = route_stops_for_route[i].station_id
            s2 = route_stops_for_route[i + 1].station_id
            if s1 in station_idx and s2 in station_idx:
                A[station_idx[s1], station_idx[s2]] = 1.0
                A[station_idx[s2], station_idx[s1]] = 1.0
    D = np.diag(1.0 / (A.sum(axis=1) + 1e-8))
    A_norm = D @ A @ D
    stop_ids = [s.stop_id for s in stations]
    return A_norm, stop_ids, station_idx


def build_feature_tensor(
    session: Session,
    station_idx: dict[str, int],
    stop_ids: list[str],
    start_time: datetime,
    window_hours: int = 168,
    horizon_hours: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature tensor (1, T, N, F=16) from DB matching training layout.

    Feature indices match main.py training pipeline:
      0: passengers_boarding  1: passengers_alighting  2: load
      3: temperature          4: precipitation         5: is_holiday
      6: rush_hour            7: delta_h               8: hour_sin
      9: hour_cos            10: dow_sin              11: dow_cos
     12: roll_6h             13: roll_24h             14: dev_24h
     15: ratio_24h
    """
    N = len(stop_ids)
    begin_time = start_time - timedelta(hours=window_hours)
    target_end = start_time + timedelta(hours=horizon_hours)

    # Query historical data
    rows = (
        session.query(HistoricalRidershipORM)
        .filter(HistoricalRidershipORM.timestamp >= begin_time)
        .filter(HistoricalRidershipORM.timestamp < target_end)
        .order_by(HistoricalRidershipORM.timestamp)
        .all()
    )

    # Query weather
    weather_rows = (
        session.query(WeatherReadingORM)
        .filter(WeatherReadingORM.timestamp >= begin_time)
        .filter(WeatherReadingORM.timestamp < target_end)
        .all()
    )
    weather_map = {w.timestamp.replace(tzinfo=UTC): w for w in weather_rows}

    T_window = window_hours
    T_horizon = horizon_hours
    F = 16

    x_data = np.zeros((T_window, N, F), dtype=np.float32)
    y_data = np.zeros((T_horizon, N), dtype=np.float32)

    # Index data by (timestamp, station_id)
    data_index = {}
    for row in rows:
        ts = row.timestamp.replace(tzinfo=UTC) if row.timestamp.tzinfo is None else row.timestamp
        data_index[(ts, row.station_id)] = row

    # --- Pass 1: Fill base features (0-11) ---
    for t in range(T_window):
        ts = begin_time + timedelta(hours=t)
        w = weather_map.get(ts)
        is_hol = ts.weekday() >= 5 or (ts.month, ts.day) in KAZAKH_HOLIDAYS
        rush = 1.0 if ts.hour in (7, 8, 9, 17, 18, 19) else 0.0
        delta_h = _next_holiday_hours(ts)

        for n, sid in enumerate(stop_ids):
            row = data_index.get((ts, sid))
            if row:
                x_data[t, n, 0] = row.passengers_boarding
                x_data[t, n, 1] = row.passengers_alighting
                x_data[t, n, 2] = row.load
            if w:
                x_data[t, n, 3] = w.temperature_c or 0.0
                x_data[t, n, 4] = w.precipitation_mm or 0.0
            x_data[t, n, 5] = 1.0 if is_hol else 0.0
            x_data[t, n, 6] = rush
            x_data[t, n, 7] = delta_h
            x_data[t, n, 8] = np.sin(2 * np.pi * ts.hour / 24)
            x_data[t, n, 9] = np.cos(2 * np.pi * ts.hour / 24)
            x_data[t, n, 10] = np.sin(2 * np.pi * ts.weekday() / 7)
            x_data[t, n, 11] = np.cos(2 * np.pi * ts.weekday() / 7)

    # --- Pass 2: Compute rolling statistics (12-15) per station ---
    load_col = x_data[:, :, 2]  # (T, N)
    for n in range(N):
        series = load_col[:, n]
        # 6-hour rolling mean
        roll6 = np.zeros(T_window, dtype=np.float32)
        for t in range(T_window):
            start_idx = max(0, t - 5)
            roll6[t] = np.mean(series[start_idx : t + 1])
        # 24-hour rolling mean
        roll24 = np.zeros(T_window, dtype=np.float32)
        for t in range(T_window):
            start_idx = max(0, t - 23)
            roll24[t] = np.mean(series[start_idx : t + 1])

        x_data[:, n, 12] = roll6
        x_data[:, n, 13] = roll24
        x_data[:, n, 14] = series - roll24  # dev_24h
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(roll24 > 1e-6, series / roll24, 1.0)
        x_data[:, n, 15] = ratio.astype(np.float32)

    # --- Fill target tensor ---
    for t in range(T_horizon):
        ts = start_time + timedelta(hours=t)
        for n, sid in enumerate(stop_ids):
            row = data_index.get((ts, sid))
            if row:
                y_data[t, n] = row.passengers_boarding

    x_tensor = x_data[np.newaxis, :, :, :]  # (1, T, N, F)
    y_tensor = y_data[np.newaxis, :, :]  # (1, H, N)

    return x_tensor, y_tensor


def get_latest_data_snapshot(session: Session, hours: int = 168) -> dict:
    """Get latest data snapshot for prediction generation."""
    now = datetime.now(UTC)
    begin = now - timedelta(hours=hours)

    ridership = session.query(HistoricalRidershipORM).filter(HistoricalRidershipORM.timestamp >= begin).all()
    weather = session.query(WeatherReadingORM).filter(WeatherReadingORM.timestamp >= begin).all()

    return {
        "ridership": [
            {
                "station_id": r.station_id,
                "route_id": r.route_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "passengers_boarding": r.passengers_boarding,
                "passengers_alighting": r.passengers_alighting,
                "load": r.load,
                "weather_code": r.weather_code,
                "temperature": r.temperature,
                "is_holiday": r.is_holiday,
                "is_event_day": r.is_event_day,
                "day_of_week": r.day_of_week,
                "hour": r.hour,
            }
            for r in ridership
        ],
        "weather": [
            {
                "timestamp": w.timestamp.isoformat() if w.timestamp else "",
                "temperature_c": w.temperature_c,
                "humidity_pct": w.humidity_pct,
                "wind_speed_kmh": w.wind_speed_kmh,
                "precipitation_mm": w.precipitation_mm,
                "weather_code": w.weather_code,
                "description": w.description,
                "is_forecast": w.is_forecast,
            }
            for w in weather
        ],
        "hours": hours,
    }
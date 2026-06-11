"""Timeline API — continuous actual vs predicted ridership time series."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models_orm import ForecastORM, HistoricalRidershipORM

router = APIRouter()

RESOLUTION_DELTAS = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
}


@router.get("")
def get_timeline(
    station_id: str | None = Query(None, description="Filter to a single station"),
    start_time: datetime = Query(..., description="Start of the time window (ISO 8601)"),
    end_time: datetime = Query(..., description="End of the time window (ISO 8601)"),
    resolution: str = Query("15m", description="Time bucket size: 5m, 15m, or 1h"),
    db: Session = Depends(get_db_session),
):
    """Return a continuous time series of actual and predicted ridership.

    Past data points have non-null `actual` from historical records.
    Future data points have `actual=null` and `predicted` from forecasts.
    """
    delta = RESOLUTION_DELTAS.get(resolution)
    if delta is None:
        return {"error": f"Invalid resolution '{resolution}'. Use 5m, 15m, or 1h."}

    now = datetime.now(UTC)

    # Ensure datetimes are timezone-aware (treat naive as UTC)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=UTC)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=UTC)

    # Build the list of bucket timestamps
    buckets = []
    t = start_time
    while t < end_time:
        buckets.append(t)
        t += delta

    # Fetch historical ridership for past buckets
    hist_query = db.query(HistoricalRidershipORM).filter(
        HistoricalRidershipORM.timestamp >= start_time,
        HistoricalRidershipORM.timestamp < end_time,
    )
    if station_id:
        hist_query = hist_query.filter(HistoricalRidershipORM.station_id == station_id)
    historical = hist_query.all()

    # Index historical records by (timestamp_truncated_to_hour, station_id)
    # Normalize to naive datetimes for consistent dict key matching
    # (SQLite stores naive, bucket timestamps may be timezone-aware)
    def _to_naive(dt):
        """Strip timezone info for consistent dict key comparison."""
        if dt is None:
            return None
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    hist_by_bucket: dict = {}
    for h in historical:
        # Snap to nearest hour for matching
        h_key = (_to_naive(h.timestamp.replace(minute=0, second=0, microsecond=0)), h.station_id)
        hist_by_bucket.setdefault(h_key, []).append(h)

    # Fetch forecasts for future buckets
    fc_query = db.query(ForecastORM).filter(
        ForecastORM.timestamp >= start_time,
        ForecastORM.timestamp < end_time,
    )
    if station_id:
        fc_query = fc_query.filter(ForecastORM.station_id == station_id)
    forecasts = fc_query.all()

    # Index forecast records by (timestamp, station_id)
    fc_by_bucket: dict = {}
    for f in forecasts:
        f_key = (_to_naive(f.timestamp), f.station_id)
        fc_by_bucket[f_key] = f
        # Also index by hour-snapped key for wider matching
        f_hour_key = (_to_naive(f.timestamp.replace(minute=0, second=0, microsecond=0)), f.station_id)
        if f_hour_key not in fc_by_bucket:
            fc_by_bucket[f_hour_key] = f

    # If no station_id specified, get all stations that have data
    if station_id:
        station_ids = [station_id]
    else:
        station_ids = list({h.station_id for h in historical} | {f.station_id for f in forecasts})
        # If no data at all, use all known stations
        if not station_ids:
            from backend.models_orm import StationORM

            station_ids = [s.stop_id for s in db.query(StationORM).all()]

    # Build the timeline series
    series = []
    for bucket_ts in buckets:
        is_past = bucket_ts <= now
        bucket_hour_naive = _to_naive(bucket_ts.replace(minute=0, second=0, microsecond=0))

        for sid in station_ids:
            entry = {"timestamp": bucket_ts.isoformat(), "station_id": sid}

            # Actual value from historical data (past only)
            if is_past:
                h_key = (bucket_hour_naive, sid)
                h_records = hist_by_bucket.get(h_key, [])
                if h_records:
                    # Average passengers_boarding across records in this bucket
                    entry["actual"] = sum(r.passengers_boarding for r in h_records) / len(h_records)
                else:
                    entry["actual"] = None
            else:
                entry["actual"] = None

            # Predicted value from forecasts
            # Try exact timestamp match first, then hour-snapped match
            fc = fc_by_bucket.get((_to_naive(bucket_ts), sid)) or fc_by_bucket.get((bucket_hour_naive, sid))
            if fc:
                entry["predicted"] = fc.predicted
                confidence = fc.confidence or 0.8
                half_width = (1 - confidence) * fc.predicted
                entry["confidence_upper"] = round(fc.predicted + half_width, 2)
                entry["confidence_lower"] = round(max(0, fc.predicted - half_width), 2)
            else:
                entry["predicted"] = None
                entry["confidence_upper"] = None
                entry["confidence_lower"] = None

            series.append(entry)

    return {
        "timeline": series,
        "resolution": resolution,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "station_id": station_id,
        "total_points": len(series),
    }

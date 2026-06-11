"""Simulation API — start/stop simulation, query state and metrics."""

import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models_orm import PredictionAccuracyORM
from backend.redis_client import get_redis

router = APIRouter()
logger = logging.getLogger(__name__)

# Redis keys used by the simulation engine (backend/tasks.py)
TASK_ID_KEY = "michi:sim:task_id"
CHECKPOINT_KEY = "michi:simulation:checkpoint"
METRICS_HISTORY_KEY = "michi:simulation:metrics_history"
STATION_DATA_KEY = "michi:simulation:latest_station_data"


@router.post("/start", status_code=202)
def start_simulation():
    """Trigger the simulation Celery task. Returns 202 with task_id.

    Uses a Redis SETNX lock to prevent race conditions on concurrent starts.
    """
    from backend.tasks import celery_app

    r = get_redis()
    # Prevent concurrent starts with a 30-second lock
    lock_key = "michi:sim:start_lock"
    acquired = r.set(lock_key, "1", nx=True, ex=30)
    if not acquired:
        existing_id = r.get(TASK_ID_KEY)
        return {"status": "already_running", "task_id": existing_id}

    try:
        existing_id = r.get(TASK_ID_KEY)
        if existing_id:
            result = celery_app.AsyncResult(existing_id)
            if result.state in ("PENDING", "STARTED", "RETRY"):
                return {"status": "already_running", "task_id": existing_id}

        result = celery_app.send_task("run_simulation")
        task_id = result.id
        r.set(TASK_ID_KEY, task_id)
        return {"status": "started", "task_id": task_id}
    finally:
        r.delete(lock_key)


@router.post("/stop")
def stop_simulation():
    """Revoke the running simulation Celery task."""
    from backend.tasks import celery_app

    r = get_redis()
    task_id = r.get(TASK_ID_KEY)
    if task_id:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
        r.delete(TASK_ID_KEY)

    return {"status": "stopped", "task_id": task_id}


@router.get("/state")
def get_simulation_state():
    """Return current simulation state: running, tick, metrics, drift status."""
    from backend.tasks import celery_app

    r = get_redis()
    task_id = r.get(TASK_ID_KEY)
    running = False
    if task_id:
        result = celery_app.AsyncResult(task_id)
        running = result.state in ("PENDING", "STARTED", "RETRY")

    # Read checkpoint from simulation engine
    checkpoint = {}
    try:
        raw = r.get(CHECKPOINT_KEY)
        if raw:
            checkpoint = json.loads(raw)
    except Exception:
        logger.warning("Failed to read simulation checkpoint from Redis")

    # Read latest metrics
    latest_metrics = None
    try:
        raw = r.get(METRICS_HISTORY_KEY)
        if raw:
            history = json.loads(raw)
            if history:
                latest_metrics = history[-1]
    except Exception:
        logger.warning("Failed to read simulation metrics from Redis")

    # Derive station_count from checkpoint or station data
    station_count = checkpoint.get("station_count")
    if station_count is None:
        try:
            raw_sd = r.get(STATION_DATA_KEY)
            if raw_sd:
                station_count = len(json.loads(raw_sd))
        except Exception:
            logger.warning("Failed to read station data from Redis")

    return {
        "running": running,
        "task_id": task_id,
        "tick": checkpoint.get("tick_count", 0),
        "current_time": checkpoint.get("current_time"),
        "drift_status": checkpoint.get("drift_status", "normal"),
        "metrics": {
            "mae": latest_metrics.get("mae") if latest_metrics else None,
            "mape": latest_metrics.get("mape") if latest_metrics else None,
            "accuracy": latest_metrics.get("accuracy") if latest_metrics else None,
        }
        if latest_metrics
        else {"mae": None, "mape": None, "accuracy": None},
        "station_count": station_count,
    }


@router.get("/metrics")
def get_simulation_metrics(hours_back: int = 24, db: Session = Depends(get_db_session)):
    """Return historical MAE/MAPE time series."""
    r = get_redis()
    realtime_metrics = []
    try:
        raw = r.get(METRICS_HISTORY_KEY)
        if raw:
            realtime_metrics = json.loads(raw)
    except Exception:
        logger.warning("Failed to read realtime metrics from Redis")

    # DB-stored prediction accuracy records
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=hours_back)
    records = (
        db.query(PredictionAccuracyORM)
        .filter(PredictionAccuracyORM.evaluated_at >= cutoff)
        .order_by(PredictionAccuracyORM.evaluated_at)
        .all()
    )

    hourly: dict = {}
    for rec in records:
        key = rec.evaluated_at.strftime("%Y-%m-%dT%H:00") if rec.evaluated_at else "unknown"
        hourly.setdefault(key, {"abs_errors": [], "mape_vals": []})
        if rec.absolute_error is not None:
            hourly[key]["abs_errors"].append(float(rec.absolute_error))
        if rec.mape is not None:
            hourly[key]["mape_vals"].append(float(rec.mape))

    db_metrics = []
    for ts in sorted(hourly.keys()):
        ae = hourly[ts]["abs_errors"]
        mp = hourly[ts]["mape_vals"]
        db_metrics.append(
            {
                "timestamp": ts,
                "mae": round(sum(ae) / len(ae), 2) if ae else None,
                "mape": round(sum(mp) / len(mp) * 100, 2) if mp else None,
                "count": len(ae) + len(mp),
            }
        )

    return {
        "realtime": realtime_metrics[-100:],
        "database": db_metrics,
        "hours_back": hours_back,
    }


@router.get("/station-data")
def get_station_data():
    """Return latest per-station simulation data."""
    r = get_redis()
    try:
        raw = r.get(STATION_DATA_KEY)
        if raw:
            return {"stations": json.loads(raw), "updated_at": datetime.now(UTC).isoformat()}
    except Exception:
        logger.warning("Failed to read station data from Redis")
    return {"stations": {}, "updated_at": None}

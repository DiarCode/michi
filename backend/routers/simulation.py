"""Simulation API — start/stop simulation, query state and metrics."""
import json
import os
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models_orm import PredictionAccuracyORM

router = APIRouter()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis keys used by the simulation engine (backend/tasks.py)
TASK_ID_KEY = "michi:sim:task_id"
CHECKPOINT_KEY = "michi:simulation:checkpoint"
METRICS_HISTORY_KEY = "michi:simulation:metrics_history"
STATION_DATA_KEY = "michi:simulation:latest_station_data"


def _redis():
    """Return a synchronous Redis client."""
    import redis as _redis
    return _redis.from_url(REDIS_URL, socket_connect_timeout=2)


def _get_task_id() -> str | None:
    """Read the currently tracked simulation task ID from Redis."""
    try:
        return _redis().get(TASK_ID_KEY)
    except Exception:
        return None


def _set_task_id(task_id: str | None):
    """Store or clear the simulation task ID in Redis."""
    try:
        r = _redis()
        if task_id:
            r.set(TASK_ID_KEY, task_id)
        else:
            r.delete(TASK_ID_KEY)
    except Exception:
        pass


@router.post("/start", status_code=202)
def start_simulation():
    """Trigger the simulation Celery task. Returns 202 with task_id."""
    from backend.tasks import celery_app

    existing_id = _get_task_id()
    if existing_id:
        # Check if the task is still running
        result = celery_app.AsyncResult(existing_id)
        if result.state in ("PENDING", "STARTED", "RETRY"):
            return {"status": "already_running", "task_id": existing_id}

    result = celery_app.send_task("run_simulation")
    task_id = result.id
    _set_task_id(task_id)

    return {"status": "started", "task_id": task_id}


@router.post("/stop")
def stop_simulation():
    """Revoke the running simulation Celery task."""
    from backend.tasks import celery_app

    task_id = _get_task_id()
    if task_id:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    _set_task_id(None)

    return {"status": "stopped", "task_id": task_id}


@router.get("/state")
def get_simulation_state():
    """Return current simulation state: running, tick, metrics, drift status.

    Reads the checkpoint written by the simulation engine every 60 ticks
    and merges it with the Celery task running status.
    """
    from backend.tasks import celery_app

    task_id = _get_task_id()
    running = False
    if task_id:
        result = celery_app.AsyncResult(task_id)
        running = result.state in ("PENDING", "STARTED", "RETRY")

    # Read checkpoint from simulation engine
    checkpoint = {}
    try:
        raw = _redis().get(CHECKPOINT_KEY)
        if raw:
            checkpoint = json.loads(raw)
    except Exception:
        pass

    # Read latest metrics from the in-Redis history
    latest_metrics = None
    try:
        raw = _redis().get(METRICS_HISTORY_KEY)
        if raw:
            history = json.loads(raw)
            if history:
                latest_metrics = history[-1]
    except Exception:
        pass

    # Derive station_count from checkpoint or station data in Redis
    station_count = checkpoint.get("station_count")
    if station_count is None:
        try:
            raw_sd = _redis().get(STATION_DATA_KEY)
            if raw_sd:
                station_count = len(json.loads(raw_sd))
        except Exception:
            pass

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
        } if latest_metrics else {"mae": None, "mape": None, "accuracy": None},
        "station_count": station_count,
    }


@router.get("/metrics")
def get_simulation_metrics(hours_back: int = 24, db: Session = Depends(get_db_session)):
    """Return historical MAE/MAPE time series.

    Combines real-time metrics from Redis (from the simulation engine)
    with persisted prediction accuracy records from the database.
    """
    # Real-time metrics from simulation engine (stored in Redis)
    realtime_metrics = []
    try:
        raw = _redis().get(METRICS_HISTORY_KEY)
        if raw:
            realtime_metrics = json.loads(raw)
    except Exception:
        pass

    # DB-stored prediction accuracy records
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=hours_back)
    records = (
        db.query(PredictionAccuracyORM)
        .filter(PredictionAccuracyORM.evaluated_at >= cutoff)
        .order_by(PredictionAccuracyORM.evaluated_at)
        .all()
    )

    # Python-side hourly aggregation (portable across SQLite/PostgreSQL)
    hourly: dict = {}
    for r in records:
        key = r.evaluated_at.strftime("%Y-%m-%dT%H:00") if r.evaluated_at else "unknown"
        hourly.setdefault(key, {"abs_errors": [], "mape_vals": []})
        if r.absolute_error is not None:
            hourly[key]["abs_errors"].append(float(r.absolute_error))
        if r.mape is not None:
            hourly[key]["mape_vals"].append(float(r.mape))

    db_metrics = []
    for ts in sorted(hourly.keys()):
        ae = hourly[ts]["abs_errors"]
        mp = hourly[ts]["mape_vals"]
        db_metrics.append({
            "timestamp": ts,
            "mae": round(sum(ae) / len(ae), 2) if ae else None,
            "mape": round(sum(mp) / len(mp) * 100, 2) if mp else None,
            "count": len(ae) + len(mp),
        })

    return {
        "realtime": realtime_metrics[-100:],  # last 100 ticks
        "database": db_metrics,
        "hours_back": hours_back,
    }


@router.get("/station-data")
def get_station_data():
    """Return latest per-station simulation data (actual, predicted, confidence)."""
    try:
        raw = _redis().get(STATION_DATA_KEY)
        if raw:
            return {"stations": json.loads(raw), "updated_at": datetime.now(UTC).isoformat()}
    except Exception:
        pass
    return {"stations": {}, "updated_at": None}

"""Celery background tasks."""

import json
import logging
import os
import time
from datetime import UTC

from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("michi", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Almaty",
    enable_utc=True,
    beat_schedule={
        "generate-forecasts": {
            "task": "backend.tasks.generate_forecasts",
            "schedule": 900.0,
        },
    },
)


def _get_db_session():
    """Create a short-lived DB session for Celery workers."""
    from backend.database import SessionLocal
    return SessionLocal()


@celery_app.task
def generate_forecasts():
    """Generate forecasts for all stations using DTS-GSSF model and persist to DB."""
    from datetime import datetime

    from backend.models_orm import ForecastORM
    from backend.services.forecast_service import generate_all_forecasts

    db = _get_db_session()
    try:
        predictions = generate_all_forecasts(db)
        if not predictions:
            return {"status": "skipped", "reason": "no predictions generated"}

        count = 0
        now = datetime.now(UTC)
        for entry in predictions:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(entry["timestamp"])
            existing = (db.query(ForecastORM)
                        .filter(ForecastORM.station_id == entry["station_id"],
                                ForecastORM.timestamp == ts)
                        .first())
            if existing:
                existing.predicted = entry["predicted"]
                existing.confidence = entry["confidence"]
                existing.model_version = entry.get("model_version", "dts-gssf")
                existing.created_at = now
            else:
                db.add(ForecastORM(
                    station_id=entry["station_id"],
                    timestamp=ts,
                    predicted=entry["predicted"],
                    confidence=entry["confidence"],
                    model_version=entry.get("model_version", "dts-gssf"),
                    created_at=now,
                ))
            count += 1
        db.commit()
        return {"status": "ok", "forecasts_generated": count}
    except Exception as e:
        db.rollback()
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()


@celery_app.task
def retrain_model():
    """Trigger model retraining (placeholder for DTS-GSSF pipeline)."""
    return {"status": "queued", "message": "Model retraining requested — connect DTS-GSSF pipeline to execute"}


@celery_app.task(bind=True, name="run_simulation")
def run_simulation(self):
    """Run the simulation engine as a Celery task, publishing ticks to Redis."""
    import redis as _redis

    from backend.database import SessionLocal
    from backend.services.simulation_service import SimulationEngine

    r = _redis.from_url(REDIS_URL, decode_responses=True)
    db = SessionLocal()

    try:
        engine = SimulationEngine(db)
        logger.info("Simulation engine started with %d stations", len(engine.stations))

        while True:
            tick_data = engine.tick()

            # Publish simulation tick
            r.publish("michi:simulation", json.dumps({
                "type": "simulation_tick",
                "tick": tick_data["tick"],
                "timestamp": tick_data["timestamp"],
                "hour": tick_data["hour"],
                "station_count": len(tick_data["stations"]),
            }))

            # Publish validation metrics
            r.publish("michi:simulation", json.dumps(tick_data["metrics"]))

            # Store latest station-level data in Redis (for /simulation/station-data API)
            r.set("michi:simulation:latest_station_data", json.dumps(tick_data["stations"]))

            # Checkpoint every 60 ticks
            if tick_data["tick"] % 60 == 0:
                checkpoint = engine.get_checkpoint()
                r.set("michi:simulation:checkpoint", json.dumps(checkpoint))
                r.set("michi:simulation:metrics_history", json.dumps(engine.get_metrics_history()[-100:]))

            time.sleep(1)  # 1 tick per second

    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
        r.publish("michi:simulation", json.dumps({"type": "simulation_error", "error": str(e)}))
        raise
    finally:
        db.close()


@celery_app.task(name="get_simulation_state")
def get_simulation_state():
    """Get current simulation state from Redis."""
    import redis as _redis
    r = _redis.from_url(REDIS_URL, decode_responses=True)
    checkpoint = r.get("michi:simulation:checkpoint")
    metrics = r.get("michi:simulation:metrics_history")
    return {
        "checkpoint": json.loads(checkpoint) if checkpoint else None,
        "metrics": json.loads(metrics) if metrics else [],
    }

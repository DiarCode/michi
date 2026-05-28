"""Celery background tasks."""

import os
from celery import Celery

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
    from backend.models_orm import ForecastORM
    from backend.services.forecast_service import generate_all_forecasts
    from datetime import datetime, timezone

    db = _get_db_session()
    try:
        predictions = generate_all_forecasts(db)
        if not predictions:
            return {"status": "skipped", "reason": "no predictions generated"}

        count = 0
        now = datetime.now(timezone.utc)
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
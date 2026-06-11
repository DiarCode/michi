"""ML model management API — retrain, status, and model promotion endpoints."""

import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.exceptions import NotFoundException
from backend.ml.artifact_store import (
    get_production_artifact,
    get_shadow_artifact,
    promote_shadow_to_production,
)
from backend.models_orm import ModelArtifactORM
from backend.redis_client import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class RetrainRequest(BaseModel):
    """Request body for triggering a model retrain."""

    horizon: int = 4
    epochs: int = 10


class PromoteRequest(BaseModel):
    """Request body for promoting a shadow model to production."""

    shadow_version: str | None = None


@router.post("/retrain", status_code=202)
def trigger_retrain(request: RetrainRequest = RetrainRequest()):
    """Trigger a model retrain Celery task with optional config."""
    from backend.tasks import retrain_model

    result = retrain_model.delay(horizon=request.horizon, epochs=request.epochs)

    r = get_redis()
    try:
        payload = {
            "type": "retrain_triggered",
            "task_id": result.id,
            "horizon": request.horizon,
            "epochs": request.epochs,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        r.publish("michi:ml", json.dumps(payload))
    except Exception as e:
        logger.warning("Failed to publish retrain_triggered event: %s", e)

    return {
        "status": "queued",
        "task_id": result.id,
        "horizon": request.horizon,
        "epochs": request.epochs,
    }


@router.get("/status")
def ml_status(db: Session = Depends(get_db_session)):
    """Return current model status: production/shadow versions, accuracy, drift."""
    from backend.services.accuracy_service import evaluate_accuracy

    prod = get_production_artifact(db)
    shadow = get_shadow_artifact(db)

    accuracy = evaluate_accuracy(db)

    r = get_redis()
    retrain_status = None
    try:
        raw = r.get("michi:ml:retrain_status")
        if raw:
            retrain_status = json.loads(raw)
    except Exception:
        logger.warning("Failed to read retrain status from Redis")

    drift_status = "normal"
    last_drift_alert = None
    drifted_stations = []
    try:
        raw = r.get("michi:ml:drift_status")
        if raw:
            drift_data = json.loads(raw)
            drift_status = drift_data.get("status", "normal")
            last_drift_alert = drift_data.get("timestamp")
            drifted_stations = drift_data.get("drifted_stations", [])
    except Exception:
        logger.warning("Failed to read drift status from Redis")

    prod_info = None
    if prod:
        prod_info = {
            "version": prod.version,
            "created_at": prod.created_at.isoformat() if prod.created_at else None,
            "metrics": json.loads(prod.metrics_json) if prod.metrics_json else None,
            "is_production": prod.is_production,
        }

    shadow_info = None
    if shadow:
        shadow_info = {
            "version": shadow.version,
            "created_at": shadow.created_at.isoformat() if shadow.created_at else None,
            "metrics": json.loads(shadow.metrics_json) if shadow.metrics_json else None,
            "is_shadow": shadow.is_shadow,
        }

    return {
        "production": prod_info,
        "shadow": shadow_info,
        "accuracy": accuracy,
        "drift": {
            "status": drift_status,
            "last_alert": last_drift_alert,
            "drifted_stations": drifted_stations,
        },
        "retrain": retrain_status,
    }


@router.post("/promote-shadow")
def promote_shadow(request: PromoteRequest = PromoteRequest(), db: Session = Depends(get_db_session)):
    """Promote a shadow model to production."""
    shadow = get_shadow_artifact(db)
    if not shadow:
        raise NotFoundException("Shadow model", "")

    version = request.shadow_version or shadow.version

    target = db.query(ModelArtifactORM).filter(ModelArtifactORM.version == version).first()
    if not target:
        raise NotFoundException("Model artifact", version)

    promoted = promote_shadow_to_production(db, version)
    if not promoted:
        raise NotFoundException("Model artifact", version)

    # Clear predictor cache
    from backend.ml.predictor import _model_cache, _normalizer_cache

    _model_cache.clear()
    _normalizer_cache.clear()

    # Publish promotion event
    r = get_redis()
    try:
        payload = {
            "type": "model_promoted",
            "version": promoted.version,
            "previous_production": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        r.publish("michi:ml", json.dumps(payload))
    except Exception as e:
        logger.warning("Failed to publish model_promoted event: %s", e)

    logger.info("Model promoted to production: %s", promoted.version)

    return {
        "status": "promoted",
        "version": promoted.version,
        "is_production": promoted.is_production,
    }


@router.post("/check-drift")
def check_drift(auto_retrain: bool = Query(False), db: Session = Depends(get_db_session)):
    """Run drift detection on recent prediction accuracy data."""
    from backend.ml.drift_detector import DriftMonitor

    monitor = DriftMonitor(auto_retrain=auto_retrain)
    result = monitor.check_from_db(db)

    return {
        "drift_detected": result["drift_detected"],
        "drifted_stations": result["drifted_stations"],
        "total_stations_checked": result["total_stations_checked"],
        "auto_retrain": auto_retrain,
    }


@router.post("/reset-drift")
def reset_drift():
    """Reset all drift detectors and clear drift status."""
    from backend.ml.drift_detector import DriftMonitor

    monitor = DriftMonitor()
    monitor.reset()

    return {"status": "ok", "message": "Drift detectors reset"}

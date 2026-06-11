"""Celery background tasks."""

import contextlib
import json
import logging
import time
from datetime import UTC, datetime

from celery import Celery

from backend.redis_client import REDIS_URL

logger = logging.getLogger(__name__)

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
        "fetch-weather": {
            "task": "fetch_weather",
            "schedule": 1800.0,  # every 30 minutes
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
            existing = (
                db.query(ForecastORM)
                .filter(ForecastORM.station_id == entry["station_id"], ForecastORM.timestamp == ts)
                .first()
            )
            if existing:
                existing.predicted = entry["predicted"]
                existing.confidence = entry["confidence"]
                existing.model_version = entry.get("model_version", "dts-gssf")
                existing.created_at = now
            else:
                db.add(
                    ForecastORM(
                        station_id=entry["station_id"],
                        timestamp=ts,
                        predicted=entry["predicted"],
                        confidence=entry["confidence"],
                        model_version=entry.get("model_version", "dts-gssf"),
                        created_at=now,
                    )
                )
            count += 1
        db.commit()
        return {"status": "ok", "forecasts_generated": count}
    except Exception as e:
        db.rollback()
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()


def _publish_ml_event(event_type: str, data: dict):
    """Publish an ML event to the michi:ml Redis channel."""
    try:
        from backend.redis_client import get_redis

        r = get_redis()
        payload = {"type": event_type, **data, "timestamp": datetime.now(UTC).isoformat()}
        r.publish("michi:ml", json.dumps(payload))
    except Exception as e:
        logger.warning("Failed to publish ML event to Redis: %s", e)


def _store_retrain_status(task_id: str, status: str, details: dict | None = None):
    """Store retrain status in Redis for API queries."""
    try:
        from backend.redis_client import get_redis

        r = get_redis()
        data = {"task_id": task_id, "status": status, "updated_at": datetime.now(UTC).isoformat()}
        if details:
            data.update(details)
        r.set("michi:ml:retrain_status", json.dumps(data))
    except Exception as e:
        logger.warning("Failed to store retrain status in Redis: %s", e)


@celery_app.task(bind=True, name="retrain_model")
def retrain_model(self, horizon: int = 4, epochs: int = 10):
    """Retrain the DTS-GSSF model and save as shadow artifact.

    Pipeline:
    1. Load training data from DB (HistoricalRidershipORM)
    2. Build adjacency and feature tensors
    3. Initialize model (warm-start from production if available)
    4. Train for specified epochs using Negative Binomial NLL
    5. Save as shadow artifact via artifact_store
    6. Evaluate accuracy and compare shadow vs production
    7. Auto-promote shadow if accuracy improves
    8. Broadcast progress via Redis pub/sub
    """
    import numpy as np
    import torch
    import torch.optim as optim

    from backend.ml.artifact_store import (
        get_production_artifact,
        promote_shadow_to_production,
        save_artifact,
    )
    from backend.ml.data_loader import build_adjacency, build_feature_tensor
    from backend.ml.model import DTSGSSF, nb_nll
    from backend.ml.normalizer import FeatureNormalizer

    task_id = self.request.id
    _publish_ml_event("retrain_started", {"task_id": task_id, "horizon": horizon, "epochs": epochs})
    _store_retrain_status(task_id, "started", {"horizon": horizon, "epochs": epochs})

    db = _get_db_session()
    try:
        # Step 1: Build adjacency matrix and station topology
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "building_adjacency"})
        _store_retrain_status(task_id, "loading_data", {"step": "building_adjacency"})

        A_phys, stop_ids, station_idx = build_adjacency(db)
        N = len(stop_ids)
        if N == 0:
            logger.error("Retrain aborted: no stations found in database")
            _publish_ml_event("retrain_failed", {"task_id": task_id, "reason": "no_stations"})
            _store_retrain_status(task_id, "failed", {"reason": "no_stations"})
            return {"status": "error", "detail": "No stations found in database"}

        # Step 2: Build feature tensors from historical data
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "loading_data"})
        _store_retrain_status(task_id, "loading_data", {"step": "feature_tensor"})

        now = datetime.now(UTC)
        x, y = build_feature_tensor(db, station_idx, stop_ids, now)

        if x.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Retrain aborted: no training data available")
            _publish_ml_event("retrain_failed", {"task_id": task_id, "reason": "no_training_data"})
            _store_retrain_status(task_id, "failed", {"reason": "no_training_data"})
            return {"status": "error", "detail": "No training data available"}

        logger.info("Training data shape: x=%s, y=%s", x.shape, y.shape)

        # Step 3: Initialize model
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "initializing_model"})
        _store_retrain_status(task_id, "initializing", {"step": "model_init"})

        n_agg = 3
        config = {
            "F_in": 11,
            "horizon": horizon,
            "d_model": 192,
            "K": 3,
            "lora_r": 16,
            "dropout": 0.1,
            "n_heads": 6,
            "n_series": N,
            "n_agg": n_agg,
        }

        model = DTSGSSF(
            N=N,
            F_in=11,
            n_series=N,
            n_agg=n_agg,
            A_phys=A_phys,
            d_model=192,
            horizon=horizon,
            K=3,
            lora_r=16,
            dropout=0.1,
            n_heads=6,
        )

        # Fit normalizer on training features
        normalizer = FeatureNormalizer()
        normalizer.fit(x)
        logger.info("Fitted feature normalizer: %s", normalizer)

        # Warm-start from production model if available
        prod_artifact = get_production_artifact(db)
        if prod_artifact:
            try:
                state = torch.load(prod_artifact.artifact_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                logger.info("Warm-started from production model %s", prod_artifact.version)
            except Exception as e:
                logger.warning("Could not load production model weights for warm-start: %s", e)

        # Step 4: Train
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "training", "epochs": epochs})
        _store_retrain_status(task_id, "training", {"step": "training", "total_epochs": epochs})

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        # Apply normalization for training
        if normalizer.is_fitted:
            x_tensor = torch.as_tensor(normalizer.transform(x), dtype=torch.float32)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            mu, kappa = model(x_tensor)
            # Align target shape with model output
            target = y_tensor[:, : mu.shape[1], : mu.shape[2]]
            loss = nb_nll(target, mu, kappa)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))

            if (epoch + 1) % max(1, epochs // 5) == 0:
                _publish_ml_event(
                    "retrain_progress",
                    {
                        "task_id": task_id,
                        "step": "training",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "loss": float(loss),
                    },
                )

        model.eval()

        # Step 5: Evaluate on training data
        with torch.no_grad():
            mu_pred, _kappa_pred = model(x_tensor)
            pred_np = mu_pred.cpu().numpy().squeeze()
            target_np = y_tensor.cpu().numpy().squeeze()
            if pred_np.ndim >= 1 and target_np.ndim >= 1:
                mae = float(np.mean(np.abs(pred_np - target_np)))
                mape = float(np.mean(np.abs(pred_np - target_np) / (np.abs(target_np) + 1e-8)))
            else:
                mae = float(np.abs(pred_np - target_np))
                mape = mae / (abs(target_np) + 1e-8)
            rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape * 100, 2),
            "train_loss": round(train_losses[-1], 4),
            "epochs": epochs,
        }

        logger.info("Shadow model training complete: MAE=%.4f, MAPE=%.2f%%, RMSE=%.4f", mae, mape * 100, rmse)

        # Step 6: Save as shadow artifact
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "saving_artifact"})
        _store_retrain_status(task_id, "saving", {"step": "saving_artifact"})

        model_state = model.state_dict()
        dataset_hash = f"retrain-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

        # Save checkpoint with normalizer state for inference
        checkpoint = {
            "model_state_dict": model_state,
            "version": f"dts-gssf-v{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            "config": config,
            "normalizer": normalizer.state_dict(),
        }

        artifact = save_artifact(
            db=db,
            model_state=model_state,
            metrics=metrics,
            config=config,
            dataset_hash=dataset_hash,
            is_shadow=True,
            is_production=False,
        )

        # Also persist normalizer in the checkpoint file
        import pathlib

        import torch as _torch

        checkpoint_path = pathlib.Path(artifact.artifact_path)
        _torch.save(checkpoint, str(checkpoint_path))

        logger.info("Shadow model saved: %s", artifact.version)

        # Step 7: Compare shadow vs production accuracy and auto-promote
        _publish_ml_event("retrain_progress", {"task_id": task_id, "step": "evaluating_accuracy"})
        _store_retrain_status(task_id, "evaluating", {"step": "evaluating_accuracy"})

        should_promote = False
        prod_metrics = None
        if prod_artifact and prod_artifact.metrics_json:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                prod_metrics = json.loads(prod_artifact.metrics_json)

        if prod_metrics and "mape" in prod_metrics:
            if mape * 100 < prod_metrics["mape"]:
                should_promote = True
                logger.info(
                    "Shadow outperforms production (shadow MAPE=%.1f%% vs prod MAPE=%.1f%%) — auto-promoting",
                    mape * 100,
                    prod_metrics["mape"],
                )
        elif not prod_artifact:
            should_promote = True
            logger.info("No production model found — auto-promoting shadow model")

        if should_promote:
            promoted = promote_shadow_to_production(db, artifact.version)
            # Clear predictor cache so new model gets loaded
            from backend.ml.predictor import _model_cache, _normalizer_cache

            _model_cache.clear()
            _normalizer_cache.clear()
            _publish_ml_event(
                "model_promoted",
                {
                    "version": promoted.version if promoted else artifact.version,
                    "previous_production": prod_artifact.version if prod_artifact else None,
                },
            )
            logger.info("Model promoted to production: %s", promoted.version if promoted else artifact.version)

        # Step 8: Final status
        result = {
            "status": "ok",
            "version": artifact.version,
            "metrics": metrics,
            "promoted": should_promote,
        }
        _publish_ml_event("retrain_completed", result)
        _store_retrain_status(task_id, "completed", result)
        return result

    except Exception as e:
        logger.error("Retrain failed: %s", e, exc_info=True)
        _publish_ml_event("retrain_failed", {"task_id": task_id, "error": str(e)})
        _store_retrain_status(task_id, "failed", {"error": str(e)})
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()


@celery_app.task(bind=True, name="run_simulation")
def run_simulation(self):
    """Run the simulation engine as a Celery task, publishing ticks to Redis."""
    from backend.database import SessionLocal
    from backend.redis_client import get_redis
    from backend.services.simulation_service import SimulationEngine

    r = get_redis()
    db = SessionLocal()

    try:
        engine = SimulationEngine(db)
        logger.info("Simulation engine started with %d stations", len(engine.stations))

        while True:
            tick_data = engine.tick()

            # Publish simulation tick
            r.publish(
                "michi:simulation",
                json.dumps(
                    {
                        "type": "simulation_tick",
                        "tick": tick_data["tick"],
                        "timestamp": tick_data["timestamp"],
                        "hour": tick_data["hour"],
                        "station_count": len(tick_data["stations"]),
                    }
                ),
            )

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
    from backend.redis_client import get_redis

    r = get_redis()
    checkpoint = r.get("michi:simulation:checkpoint")
    metrics = r.get("michi:simulation:metrics_history")
    return {
        "checkpoint": json.loads(checkpoint) if checkpoint else None,
        "metrics": json.loads(metrics) if metrics else [],
    }


@celery_app.task(name="fetch_weather")
def fetch_weather_task():
    """Fetch current weather and short-term forecast every 30 minutes."""
    from backend.database import SessionLocal
    from backend.services.weather_service import fetch_current_weather, fetch_forecast_weather

    db = SessionLocal()
    try:
        current = fetch_current_weather(db)
        forecast = fetch_forecast_weather(db, hours=24)
        logger.info(
            "Weather fetch complete: current=%.1f°C, forecast_hours=%d",
            current.get("temperature_c", 0) or 0,
            len(forecast),
        )
        return {"status": "ok", "current_temp": current.get("temperature_c"), "forecast_count": len(forecast)}
    except Exception as e:
        logger.error("Weather fetch task failed: %s", e, exc_info=True)
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

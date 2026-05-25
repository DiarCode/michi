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

@celery_app.task
def generate_forecasts():
    print("Generating forecasts...")
    return {"status": "ok"}

@celery_app.task
def retrain_model():
    print("Retraining model...")
    return {"status": "ok"}

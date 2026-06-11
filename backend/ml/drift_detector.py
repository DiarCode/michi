"""Page-Hinkley drift detection for DTS-GSSF predictions.

Includes DriftMonitor that integrates with Redis pub/sub for
real-time drift alerts and optional auto-retrain triggering.
"""

import json
import logging
from collections import deque
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


class PageHinkleyDetector:
    """Detect concept drift in prediction residuals using Page-Hinkley test.

    Monitors the running mean of residuals and triggers when cumulative
    deviation exceeds a threshold, indicating the model may need retraining.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, lambda_: float = 0.85, window: int = 500):
        self.delta = delta
        self.threshold = threshold
        self.lambda_ = lambda_
        self.window = window
        self.running_mean = 0.0
        self.running_var = 0.0
        self.n = 0
        self.cumulative_sum = 0.0
        self.min_sum = 0.0
        self.recent_residuals: deque = deque(maxlen=window)
        self._drift_detected = False

    def update(self, residual: float) -> bool:
        """Add a residual and check for drift. Returns True if drift detected."""
        self.n += 1
        self.recent_residuals.append(residual)
        old_mean = self.running_mean
        self.running_mean += (residual - self.running_mean) / self.n
        if self.n > 1:
            self.running_var = self.lambda_ * self.running_var + (1 - self.lambda_) * (residual - old_mean) ** 2
        x_sum = residual - self.running_mean - self.delta
        self.cumulative_sum += x_sum
        self.min_sum = min(self.min_sum, self.cumulative_sum)
        self._drift_detected = (self.cumulative_sum - self.min_sum) > self.threshold
        return self._drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def recent_mape(self) -> float:
        """Compute recent MAPE from buffered residuals."""
        if not self.recent_residuals:
            return 0.0
        return sum(abs(r) for r in self.recent_residuals) / len(self.recent_residuals)

    def reset(self):
        """Reset detector state after retraining."""
        self.running_mean = 0.0
        self.running_var = 0.0
        self.n = 0
        self.cumulative_sum = 0.0
        self.min_sum = 0.0
        self.recent_residuals.clear()
        self._drift_detected = False


class DriftManager:
    """Manage drift detection across multiple routes/stations."""

    def __init__(self, delta: float = 0.005, threshold: float = 50.0):
        self.delta = delta
        self.threshold = threshold
        self.detectors: dict = {}

    def get_detector(self, key: str) -> PageHinkleyDetector:
        if key not in self.detectors:
            self.detectors[key] = PageHinkleyDetector(self.delta, self.threshold)
        return self.detectors[key]

    def check_drift(self, key: str, residual: float) -> bool:
        return self.get_detector(key).update(residual)

    def get_drifted_keys(self) -> list[str]:
        return [k for k, d in self.detectors.items() if d.drift_detected]

    def reset_all(self):
        for d in self.detectors.values():
            d.reset()


class DriftMonitor:
    """Monitor drift across stations and broadcast alerts via Redis pub/sub.

    Integrates PageHinkleyDetector with the ML pipeline:
    - Feeds prediction residuals to per-station detectors
    - Publishes drift_alert events to Redis (michi:ml channel)
    - Optionally auto-triggers model retraining via Celery
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, auto_retrain: bool = False):
        self.manager = DriftManager(delta=delta, threshold=threshold)
        self.auto_retrain = auto_retrain

    def _get_redis(self):
        """Return a Redis client using the shared connection pool."""
        from backend.redis_client import get_redis

        return get_redis()

    def _publish(self, event_type: str, data: dict):
        """Publish an ML event to the michi:ml Redis channel."""
        try:
            r = self._get_redis()
            payload = {"type": event_type, **data, "timestamp": datetime.now(UTC).isoformat()}
            r.publish("michi:ml", json.dumps(payload))
        except Exception as e:
            logger.warning("Failed to publish drift event to Redis: %s", e)

    def _store_drift_status(self, status: str, details: dict | None = None):
        """Store drift status in Redis for API queries."""
        try:
            r = self._get_redis()
            data = {"status": status, "timestamp": datetime.now(UTC).isoformat()}
            if details:
                data.update(details)
            r.set("michi:ml:drift_status", json.dumps(data))
        except Exception as e:
            logger.warning("Failed to store drift status in Redis: %s", e)

    def check_residuals(self, residuals: dict[str, float]) -> dict:
        """Check drift from a dict of station_id -> residual values.

        Args:
            residuals: Mapping of station IDs to prediction residuals
                       (actual - predicted).

        Returns:
            dict with drift status including which stations drifted.
        """
        drifted_keys = []
        for key, residual in residuals.items():
            if self.manager.check_drift(key, residual):
                drifted_keys.append(key)
                logger.info("Drift detected for station: %s (residual=%.4f)", key, residual)

        result = {
            "drift_detected": len(drifted_keys) > 0,
            "drifted_stations": drifted_keys,
            "total_stations_checked": len(residuals),
        }

        if result["drift_detected"]:
            self._handle_drift(result)

        return result

    def _handle_drift(self, drift_result: dict):
        """Handle detected drift: broadcast alert and optionally trigger retrain."""
        alert_data = {
            "drifted_stations": drift_result["drifted_stations"],
            "total_checked": drift_result["total_stations_checked"],
            "recent_mape": {
                k: round(self.manager.get_detector(k).recent_mape(), 4) for k in drift_result["drifted_stations"]
            },
        }

        self._publish("drift_alert", alert_data)
        self._store_drift_status("critical", alert_data)

        if self.auto_retrain:
            logger.info("Auto-retrain enabled — triggering retrain_model task")
            try:
                from backend.tasks import retrain_model

                retrain_model.delay(horizon=4, epochs=10)
                self._publish(
                    "retrain_triggered",
                    {
                        "reason": "drift_detected",
                        "source": "auto",
                    },
                )
            except Exception as e:
                logger.error("Failed to trigger auto-retrain: %s", e)

    def check_from_db(self, db_session) -> dict:
        """Check drift using recent prediction accuracy records from the database.

        Computes residuals from stored prediction vs actual values and
        feeds them to per-station PageHinkley detectors.

        Args:
            db_session: SQLAlchemy session for querying PredictionAccuracyORM

        Returns:
            dict with drift status information.
        """
        from backend.models_orm import PredictionAccuracyORM

        cutoff = datetime.now(UTC) - timedelta(hours=24)
        records = (
            db_session.query(PredictionAccuracyORM)
            .filter(PredictionAccuracyORM.evaluated_at >= cutoff)
            .filter(PredictionAccuracyORM.actual.isnot(None))
            .order_by(PredictionAccuracyORM.evaluated_at)
            .all()
        )

        if not records:
            self._store_drift_status("normal", {"checked": 0})
            return {"drift_detected": False, "drifted_stations": [], "total_stations_checked": 0}

        residuals = {}
        for r in records:
            key = r.station_id
            residual = float(abs(r.predicted - r.actual))
            residuals[key] = residual

        return self.check_residuals(residuals)

    def reset(self):
        """Reset all detectors and clear drift status."""
        self.manager.reset_all()
        self._store_drift_status("normal", {"reset_at": datetime.now(UTC).isoformat()})

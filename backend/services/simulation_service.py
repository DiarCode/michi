"""Simulation engine for generating realistic passenger flow data and running model validation."""

import json
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session


class SimulationEngine:
    """Generates realistic passenger flow simulation data and runs model validation."""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.tick_count = 0
        self.current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.stations = self._load_stations()
        self.routes = self._load_routes()
        self.metrics_history = []  # [{tick, mae, mape, accuracy, timestamp}]
        self.drift_status = "normal"  # normal, warning, critical
        self.last_predictions = {}
        self.last_actuals = {}

    def _load_stations(self):
        """Load all stations from DB."""
        from backend.models_orm import StationORM
        stations = self.db.query(StationORM).all()
        return [
            {"stop_id": s.stop_id, "name": s.name, "ridership_24h": s.ridership_24h or 1000}
            for s in stations
        ]

    def _load_routes(self):
        """Load all routes from DB."""
        from backend.models_orm import RouteORM
        routes = self.db.query(RouteORM).all()
        return [
            {"route_id": r.route_id, "name": r.name, "avg_ridership": r.avg_ridership or 500}
            for r in routes
        ]

    def _generate_ridership(self, station: dict, hour: int) -> int:
        """Generate realistic ridership for a station at a given hour using sinusoidal patterns."""
        base = station["ridership_24h"] / 24

        # Rush hour factors (8am and 6pm peaks)
        morning_peak = 2.5 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        evening_peak = 2.2 * np.exp(-0.5 * ((hour - 18) / 1.5) ** 2)
        lunch_bump = 1.3 * np.exp(-0.5 * ((hour - 12) / 1.0) ** 2)
        hour_factor = 1.0 + morning_peak + evening_peak + lunch_bump

        # Weekend reduction
        is_weekend = self.current_time.weekday() >= 5
        if is_weekend:
            hour_factor *= 0.6

        # Seasonal factor (winter +15%)
        month = self.current_time.month
        seasonal = 1.15 if month in [11, 12, 1, 2] else 1.0

        # Random noise (+-10%)
        noise = 1.0 + np.random.normal(0, 0.1)
        noise = max(0.5, min(1.5, noise))  # clamp

        ridership = int(base * hour_factor * seasonal * noise)
        return max(1, ridership)

    def _generate_forecast(self, station: dict, hour: int) -> dict:
        """Generate forecast for a station (simulates model prediction with slight bias)."""
        actual = self._generate_ridership(station, hour)
        # Add slight forecast bias (simulates model imprecision)
        bias = 1.0 + np.random.normal(0, 0.05)  # +-5% bias
        predicted = max(1, int(actual * bias))
        confidence = max(0.6, min(0.99, 0.95 - abs(bias - 1.0) * 2))
        return {
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "confidence_upper": int(predicted * (1 + (1 - confidence))),
            "confidence_lower": int(predicted * confidence),
        }

    def tick(self) -> dict:
        """Generate one simulation tick with data for all stations."""
        self.tick_count += 1
        hour = self.current_time.hour

        station_data = {}
        total_abs_error = 0
        total_pct_error = 0
        total_stations = len(self.stations)

        for station in self.stations:
            forecast = self._generate_forecast(station, hour)
            station_data[station["stop_id"]] = {
                "name": station["name"],
                "actual": forecast["actual"],
                "predicted": forecast["predicted"],
                "confidence": forecast["confidence"],
                "confidence_upper": forecast["confidence_upper"],
                "confidence_lower": forecast["confidence_lower"],
            }
            abs_error = abs(forecast["predicted"] - forecast["actual"])
            pct_error = abs_error / max(1, forecast["actual"])
            total_abs_error += abs_error
            total_pct_error += pct_error

            self.last_predictions[station["stop_id"]] = forecast["predicted"]
            self.last_actuals[station["stop_id"]] = forecast["actual"]

        mae = total_abs_error / max(1, total_stations)
        mape = (total_pct_error / max(1, total_stations)) * 100
        accuracy = max(0, 100 - mape)

        # Drift detection
        if mape > 15:
            self.drift_status = "critical"
        elif mape > 10:
            self.drift_status = "warning"
        else:
            self.drift_status = "normal"

        metrics = {
            "tick": self.tick_count,
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "accuracy": round(accuracy, 2),
            "drift_status": self.drift_status,
            "timestamp": self.current_time.isoformat(),
        }
        self.metrics_history.append(metrics)

        result = {
            "type": "simulation_tick",
            "tick": self.tick_count,
            "timestamp": self.current_time.isoformat(),
            "hour": hour,
            "stations": station_data,
            "metrics": {
                "type": "validation_metric",
                **metrics,
            },
        }

        # Advance simulation time
        self.current_time += timedelta(minutes=15)

        return result

    def get_state(self) -> dict:
        """Get current simulation state."""
        return {
            "running": True,
            "tick_count": self.tick_count,
            "current_time": self.current_time.isoformat(),
            "drift_status": self.drift_status,
            "latest_metrics": self.metrics_history[-1] if self.metrics_history else None,
            "station_count": len(self.stations),
        }

    def get_metrics_history(self) -> list:
        """Get historical metrics."""
        return self.metrics_history

    def get_checkpoint(self) -> dict:
        """Get state for checkpointing."""
        return {
            "tick_count": self.tick_count,
            "current_time": self.current_time.isoformat(),
            "drift_status": self.drift_status,
            "metrics_count": len(self.metrics_history),
        }
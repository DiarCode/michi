"""Page-Hinkley drift detection for DTS-GSSF predictions."""
from collections import deque
from typing import List, Optional


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

    def get_drifted_keys(self) -> List[str]:
        return [k for k, d in self.detectors.items() if d.drift_detected]

    def reset_all(self):
        for d in self.detectors.values():
            d.reset()

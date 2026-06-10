"""Online residual correction via Kalman filter for DTS-GSSF predictions."""
from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    d_r: int = 16
    F_decay: float = 0.92
    q: float = 0.06
    ph_delta: float = 0.005
    ph_lambda: float = 0.85
    adapt_window: int = 192
    adapt_steps: int = 18
    adapt_lr: float = 8e-3
    adapt_weight_decay: float = 1e-4
    beta: float = 0.005
    r_scale: float = 1.0


class ResidualKalman:
    """Fast-timescale Kalman filter for online residual correction.

    Maintains a low-dimensional state that tracks prediction residuals,
    allowing quick corrections to model outputs based on recent observations.
    """
    def __init__(self, n_series: int, cfg: KalmanConfig = None, seed: int = 0):
        self.cfg = cfg or KalmanConfig()
        rng = np.random.default_rng(seed + 999)
        P = rng.normal(0.0, 1.0, size=(self.cfg.d_r, n_series)).astype(np.float32)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        self.P = P
        self.PT = P.T
        self.F = np.eye(self.cfg.d_r, dtype=np.float32) * self.cfg.F_decay
        self.Q = np.eye(self.cfg.d_r, dtype=np.float32) * self.cfg.q
        self.R = np.eye(self.cfg.d_r, dtype=np.float32) * self.cfg.r_scale
        self.e = np.zeros((self.cfg.d_r,), dtype=np.float32)
        self.Sigma = np.eye(self.cfg.d_r, dtype=np.float32) * 1.0

    def predict(self) -> np.ndarray:
        """Predict residual correction for next timestep."""
        self.e = self.F @ self.e
        self.Sigma = self.F @ self.Sigma @ self.F.T + self.Q
        return (self.PT @ self.e).astype(np.float32)

    def update(self, residual: np.ndarray) -> np.ndarray:
        """Update state with observed residual."""
        r_tilde = (self.P @ residual).astype(np.float32)
        S = self.Sigma + self.R
        Sinv = np.linalg.inv(S.astype(np.float64)).astype(np.float32)
        K = self.Sigma @ Sinv
        innov = r_tilde - self.e
        self.e = self.e + K @ innov
        self.Sigma = (np.eye(self.cfg.d_r, dtype=np.float32) - K) @ self.Sigma
        return (self.PT @ self.e).astype(np.float32)

    def correct(self, prediction: np.ndarray, residual: np.ndarray = None) -> np.ndarray:
        """Apply Kalman correction to a prediction.

        If residual is provided (observation available), updates state first.
        Returns corrected prediction.
        """
        correction = self.predict()
        if residual is not None:
            correction = self.update(residual)
        return prediction + correction

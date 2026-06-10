"""Z-score feature normalizer — ported from main.py for backend inference.

Computes per-feature mean and std on training data, then applies
(X - mean) / std to all inputs. Stats are saved in model checkpoints
and must be loaded for inference-time normalization.
"""

import numpy as np


class FeatureNormalizer:
    """Z-score normalizer fit on training data only.

    Stores per-feature mean and std for reproducible normalization
    at inference time. Compatible with checkpoint format used in main.py.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None

    def fit(self, X_train: np.ndarray) -> "FeatureNormalizer":
        """Fit on training data.

        Args:
            X_train: shape (T, N, F) or (T, F). Computes per-feature
                      mean and std over all dimensions except the last.
        """
        self.mean_ = X_train.mean(axis=tuple(range(X_train.ndim - 1)), keepdims=True)
        self.std_ = X_train.std(axis=tuple(range(X_train.ndim - 1)), keepdims=True)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization: (X - mean) / std."""
        if not self.is_fitted:
            return X
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization: X * std + mean."""
        if not self.is_fitted:
            return X
        return X * self.std_ + self.mean_

    def state_dict(self) -> dict[str, np.ndarray]:
        """Serialize normalizer state for checkpoint saving."""
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        """Restore normalizer state from checkpoint."""
        self.mean_ = d["mean"]
        self.std_ = d["std"]

    def compatible_with(self, n_features: int) -> bool:
        """Check if the normalizer's feature count matches the expected count."""
        if not self.is_fitted:
            return False
        return self.mean_.shape[-1] == n_features

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"FeatureNormalizer(fitted=True, features={self.mean_.shape[-1]})"
        return "FeatureNormalizer(fitted=False)"

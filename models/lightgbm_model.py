"""LightGBM classifier wrapper."""

from __future__ import annotations

import numpy as np

try:
    from lightgbm import LGBMClassifier

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class LightGBMModel:
    """Small LightGBM wrapper with optional sample weights."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        if not HAS_LGB:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(labels)),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(features, labels, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

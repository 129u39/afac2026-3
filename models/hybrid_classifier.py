"""Hybrid classifier: graph aggregation + LightGBM."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from lightgbm import LGBMClassifier

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class HybridClassifier:
    """LightGBM plus neighborhood label propagation."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.class_weights = None
        self.train_idx = None
        self.labels = None
        self.adj = None
        self._features = None
        self._agg_dense = None

    def fit(self, adj: csr_matrix, features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray):
        if not HAS_LGB:
            raise ImportError("lightgbm is required")

        self.adj = adj
        self.labels = labels
        self.train_idx = np.asarray(train_idx)

        deg = np.asarray(adj.sum(axis=1)).ravel()
        deg_inv = np.zeros_like(deg, dtype=np.float32)
        nonzero = deg > 0
        deg_inv[nonzero] = 1.0 / deg[nonzero]
        D_inv = diags(deg_inv)
        agg = D_inv @ adj @ features
        agg_dense = agg.toarray() if hasattr(agg, "toarray") else np.asarray(agg)
        agg_dense[~nonzero] = features[~nonzero]

        combined = np.concatenate([features, agg_dense], axis=1)
        X_train = combined[self.train_idx]
        y_train = labels[self.train_idx]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)

        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_lookup = dict(zip(classes.tolist(), class_weights.tolist()))
        sample_weights = np.asarray([weight_lookup[int(y)] for y in y_train], dtype=np.float32)
        self.class_weights = weight_lookup

        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X_train_s, y_train, sample_weight=sample_weights)

        self._features = features
        self._agg_dense = agg_dense
        return self

    def _combined_features(self, nodes: np.ndarray) -> np.ndarray:
        combined = np.concatenate([self._features, self._agg_dense], axis=1)
        return combined[nodes]

    def predict(self, test_idx: np.ndarray) -> np.ndarray:
        X_test = self._combined_features(np.asarray(test_idx))
        X_test_s = self.scaler.transform(X_test)
        lgb_proba = self.model.predict_proba(X_test_s)

        final_preds = []
        for i, node in enumerate(np.asarray(test_idx)):
            neighbors = self.adj[node].nonzero()[1]
            valid_train = np.intersect1d(neighbors, self.train_idx, assume_unique=False)

            if len(valid_train) == 0:
                final_preds.append(int(lgb_proba[i].argmax()))
                continue

            neighbor_labels = self.labels[valid_train]
            counts = Counter(neighbor_labels.tolist())
            neighbor_dist = np.zeros(lgb_proba.shape[1], dtype=np.float32)
            for cls, cnt in counts.items():
                neighbor_dist[int(cls)] = cnt / len(valid_train)

            majority_cnt = max(counts.values())
            neighbor_confidence = majority_cnt / len(valid_train)
            blend_weight = float(np.clip(0.35 + 0.55 * neighbor_confidence, 0.35, 0.9))

            blended_proba = blend_weight * neighbor_dist + (1.0 - blend_weight) * lgb_proba[i]
            final_preds.append(int(blended_proba.argmax()))

        return np.asarray(final_preds)

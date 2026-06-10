"""XGBoost 分类器。"""

import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBoostModel:
    """XGBoost 分类器。

    直接训练 XGBoost，输出类别概率用于集成。
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.05):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.is_fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """训练模型。"""
        if not HAS_XGB:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
        )
        self.model.fit(features, labels)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测类别。"""
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测类别概率。"""
        return self.model.predict_proba(features)

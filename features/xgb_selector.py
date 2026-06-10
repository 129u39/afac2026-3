"""XGBoost 特征筛选器。"""

import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBSelector:
    """XGBoost 特征筛选器。

    使用 XGBoost 的特征重要性进行特征选择。
    """

    def __init__(self, n_top_features: int = 256):
        """
        Args:
            n_top_features: 保留的特征数量
        """
        self.n_top_features = n_top_features
        self.selected_indices = None
        self.importances = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "XGBSelector":
        """拟合选择器。"""
        if not HAS_XGB:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
        )
        model.fit(features, labels)

        self.importances = model.feature_importances_
        self.selected_indices = np.argsort(self.importances)[::-1][:self.n_top_features]

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """转换特征。"""
        return features[:, self.selected_indices]

    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """拟合并转换。"""
        self.fit(features, labels)
        return self.transform(features)

    def get_selected_indices(self) -> np.ndarray:
        """获取选中的特征索引。"""
        return self.selected_indices

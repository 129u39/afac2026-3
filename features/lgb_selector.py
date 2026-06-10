"""LightGBM 特征筛选器。"""

import numpy as np

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class LGBSelector:
    """LightGBM 特征筛选器。

    使用 LightGBM 的特征重要性进行特征选择。
    """

    def __init__(self, n_top_features: int = 256):
        """
        Args:
            n_top_features: 保留的特征数量
        """
        self.n_top_features = n_top_features
        self.selected_indices = None
        self.importances = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LGBSelector":
        """拟合选择器。"""
        if not HAS_LGB:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(labels)),
            n_estimators=500,
            verbose=-1,
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

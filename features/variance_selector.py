"""低方差特征过滤器。"""

import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold


class VarianceSelector:
    """低方差特征过滤器。

    删除方差小于阈值的特征，减少噪声。
    """

    def __init__(self, threshold: float = 1e-5):
        """
        Args:
            threshold: 方差阈值
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
        self.selected_indices = None

    def fit(self, features: np.ndarray) -> "VarianceSelector":
        """拟合选择器。"""
        self.selector.fit(features)
        self.selected_indices = np.where(self.selector.get_support())[0]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """转换特征。"""
        return self.selector.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """拟合并转换。"""
        self.fit(features)
        return self.transform(features)

    def get_selected_indices(self) -> np.ndarray:
        """获取选中的特征索引。"""
        return self.selected_indices

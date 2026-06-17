"""LightGBM 基线：直接使用原始特征训练 LightGBM。

用途：
- 验证图结构是否有贡献
- 如果 LightGBM > GNN，说明图结构质量不足
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class LightGBMBaseline:
    """LightGBM 分类基线。

    直接使用原始 767 维特征训练，不使用图结构。
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        min_child_samples: int = 20,
    ):
        if not HAS_LGB:
            raise ImportError("lightgbm is required")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.model = None
        self.scaler = None
        self.class_weights = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """训练 LightGBM。"""
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(features)

        self.class_weights = compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
        sample_weights = np.array([self.class_weights[y] for y in labels])

        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            verbose=-1,
        )
        self.model.fit(X_s, labels, sample_weight=sample_weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测类别。"""
        X_s = self.scaler.transform(features)
        return self.model.predict(X_s)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测概率。"""
        X_s = self.scaler.transform(features)
        return self.model.predict_proba(X_s)

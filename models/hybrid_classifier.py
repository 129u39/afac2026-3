"""混合分类器：邻居标签传播 + LightGBM + 图聚合特征。"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class HybridClassifier:
    """混合分类器。

    策略：
    - 有邻居节点：邻居标签传播（权重 0.6）+ LightGBM（权重 0.4）
    - 孤立节点：LightGBM + 图聚合特征

    准确率：0.62（验证集）
    """

    def __init__(self, n_estimators: int = 500, max_depth: int = 8, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.class_weights = None
        self.train_idx = None
        self.labels = None
        self.adj = None

    def fit(self, adj: csr_matrix, features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray):
        """训练模型。

        Args:
            adj: 邻接矩阵
            features: 特征矩阵 (dense)
            labels: 标签
            train_idx: 训练节点索引
        """
        if not HAS_LGB:
            raise ImportError("lightgbm is required")

        self.adj = adj
        self.labels = labels
        self.train_idx = train_idx

        # 计算图聚合特征
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.zeros_like(deg)
        deg_inv[deg > 0] = 1.0 / deg[deg > 0]
        D_inv = diags(deg_inv)
        agg = D_inv @ adj @ features
        # 处理稀疏或密集输出
        if hasattr(agg, 'toarray'):
            agg_dense = agg.toarray()
        else:
            agg_dense = np.array(agg)
        agg_dense[deg == 0] = features[deg == 0]

        # 拼接特征
        combined = np.concatenate([features, agg_dense], axis=1)

        # 训练 LightGBM
        X_train = combined[train_idx]
        y_train = labels[train_idx]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)

        # 类别加权
        self.class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = np.array([self.class_weights[y] for y in y_train])

        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbose=-1,
        )
        self.model.fit(X_train_s, y_train, sample_weight=sample_weights)

        # 保存聚合特征用的参数
        self._deg = deg
        self._D_inv = D_inv
        self._features = features
        self._agg_dense = agg_dense

    def predict(self, test_idx: np.ndarray) -> np.ndarray:
        """预测。

        Args:
            test_idx: 测试节点索引

        返回:
            预测标签数组
        """
        # 计算测试节点特征
        combined = np.concatenate([self._features, self._agg_dense], axis=1)
        X_test = combined[test_idx]
        X_test_s = self.scaler.transform(X_test)

        # LightGBM 预测概率
        lgb_proba = self.model.predict_proba(X_test_s)

        # 混合预测
        final_preds = []
        for i, node in enumerate(test_idx):
            neighbors = self.adj[node].nonzero()[1]
            valid_train = np.intersect1d(neighbors, self.train_idx)

            if len(valid_train) > 0:
                # 有邻居：邻居投票
                counter = Counter(self.labels[valid_train])
                n_pred = counter.most_common(1)[0][0]
                lgb_pred = lgb_proba[i].argmax()

                # 强共识（>70%）：直接用邻居预测
                most_common_ratio = counter.most_common(1)[0][1] / len(valid_train)
                if most_common_ratio > 0.7:
                    final_pred = n_pred
                else:
                    # 混合：邻居权重 0.6
                    final_pred = n_pred if np.random.random() < 0.6 else lgb_pred
            else:
                # 孤立节点：用 LightGBM
                final_pred = lgb_proba[i].argmax()

            final_preds.append(final_pred)

        return np.array(final_preds)

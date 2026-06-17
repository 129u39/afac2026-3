"""标签传播（Label Propagation）：验证图同质性。

通过图结构传播训练标签到测试节点。
如果 LP > GCN，说明图标签传播能力极强，图结构价值主要体现在标签传播而非特征学习。
"""
import numpy as np
from scipy.sparse import diags


class LabelPropagation:
    """标签传播分类器。

    使用归一化邻接矩阵做多跳标签传播。
    只利用图结构 + 训练标签，不使用节点特征。
    """

    def __init__(self, alpha: float = 0.5, max_iter: int = 50, tol: float = 1e-4):
        """
        Args:
            alpha: 传播权重 (0~1)，越大越依赖结构传播
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.num_classes = None
        self.train_idx = None
        self.test_idx = None
        self.adj_norm = None
        self.proba = None

    def fit(self, adj_csr, labels, train_idx, test_idx):
        """训练（传播）。"""
        N = adj_csr.shape[0]
        self.num_classes = int(labels.max()) + 1
        self.train_idx = train_idx
        self.test_idx = test_idx

        # 构建归一化邻接矩阵: D^{-1/2} A D^{-1/2}
        deg = np.array(adj_csr.sum(axis=1)).flatten()
        deg_inv_sqrt = np.zeros_like(deg)
        deg_inv_sqrt[deg > 0] = np.power(deg[deg > 0], -0.5)
        D_inv_sqrt = diags(deg_inv_sqrt)
        self.adj_norm = D_inv_sqrt @ adj_csr @ D_inv_sqrt

        # 初始化标签概率矩阵 (N, C)
        Y = np.zeros((N, self.num_classes))
        for i, label in enumerate(labels):
            if label >= 0:
                Y[i, int(label)] = 1.0

        # 迭代传播
        F = Y.copy()
        for iteration in range(self.max_iter):
            F_old = F.copy()
            # F = alpha * A_norm @ F + (1 - alpha) * Y
            F = self.alpha * (self.adj_norm @ F) + (1 - self.alpha) * Y
            # 重置训练节点标签
            F[train_idx] = Y[train_idx]
            diff = np.abs(F - F_old).max()
            if diff < self.tol:
                print(f"[LP] converged at iteration={iteration}, diff={diff:.6f}")
                break

        # 归一化为概率
        row_sum = F.sum(axis=1).reshape(-1, 1)
        row_sum[row_sum == 0] = 1.0
        self.proba = F / row_sum

        print(f"[LP] alpha={self.alpha} iterations={iteration + 1}")

    def predict(self, idx=None):
        """预测类别。

        Args:
            idx: 节点索引，None 返回所有节点
        """
        if idx is None:
            idx = self.test_idx
        return self.proba[idx].argmax(axis=1)

    def predict_proba(self, idx=None):
        """预测概率。"""
        if idx is None:
            idx = self.test_idx
        return self.proba[idx]

"""SIGN: Scalable Inception Graph Neural Networks。

适合稀疏产品图场景。
结构: 预计算多跳特征 + MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SIGN(nn.Module):
    """SIGN 模型。

    预计算 A@X, A²@X, A³@X 等多跳特征，然后用 MLP 处理。
    优势：
    1. 训练速度快（不需要在训练时做图传播）
    2. 适合大规模稀疏图
    3. 可以使用 mini-batch 训练
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_hops: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_hops: 图传播跳数
            num_layers: MLP 层数
            dropout: Dropout 比率
        """
        super().__init__()
        self.num_hops = num_hops
        self.dropout = dropout

        # 每个跳数的特征投影
        self.hop_projections = nn.ModuleList()
        for _ in range(num_hops + 1):
            self.hop_projections.append(nn.Linear(in_dim, hidden_dim))

        # MLP 层
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        total_dim = hidden_dim * (num_hops + 1)
        for i in range(num_layers):
            in_c = total_dim if i == 0 else hidden_dim
            self.mlps.append(nn.Linear(in_c, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hop_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hop_features: [X0, X1, X2, ...] 其中 Xi = A^i @ X

        返回:
            logits: (num_nodes, num_classes)
        """
        # 投影每个跳数的特征
        projected = []
        for i, (feat, proj) in enumerate(zip(hop_features, self.hop_projections)):
            projected.append(proj(feat))

        # 拼接所有跳数的特征
        x = torch.cat(projected, dim=-1)

        # MLP
        for mlp, norm in zip(self.mlps, self.norms):
            x = mlp(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 分类
        x = self.classifier(x)
        return x


def precompute_sign_features(
    adj_csr,
    features: np.ndarray,
    num_hops: int = 3,
) -> list[torch.Tensor]:
    """预计算 SIGN 特征。

    Args:
        adj_csr: scipy CSR 邻接矩阵
        features: 节点特征矩阵 (num_nodes, in_dim)
        num_hops: 传播跳数

    返回:
        [X0, X1, X2, ...] 其中 Xi = A^i @ X
    """
    from scipy.sparse import csr_matrix

    # 归一化邻接矩阵: D^{-1/2} A D^{-1/2}
    adj = adj_csr.astype(np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
    D_inv_sqrt = csr_matrix(np.diag(deg_inv_sqrt))
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    # 计算多跳特征
    hop_features = []
    x = features.copy()

    for i in range(num_hops + 1):
        hop_features.append(torch.tensor(x, dtype=torch.float32))
        if i < num_hops:
            x = adj_norm @ x

    return hop_features


def train_sign(
    model: nn.Module,
    hop_features: list[torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 30,
    device: torch.device = None,
) -> dict:
    """训练 SIGN 模型。"""
    if device is None:
        device = next(model.parameters()).device

    # 移动数据到设备
    hop_features = [f.to(device) for f in hop_features]
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    train_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        out = model(hop_features)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            out = model(hop_features)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
    }

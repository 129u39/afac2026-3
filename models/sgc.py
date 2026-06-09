"""SGC: Simple Graph Convolution。

极简图卷积模型，训练速度极快。
结构: A^K X → Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SGC(nn.Module):
    """SGC 模型。

    简单图卷积：只做 K 次图传播，然后线性分类。
    优势：
    1. 训练极快（只有线性层）
    2. 稳定可靠
    3. 适合大规模产品图
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        K: int = 2,
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            K: 图传播跳数
        """
        super().__init__()
        self.K = K

        # 线性层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_dim) 预计算的 A^K X 特征

        返回:
            logits: (num_nodes, num_classes)
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def precompute_sgc_features(
    adj_csr,
    features: np.ndarray,
    K: int = 2,
) -> torch.Tensor:
    """预计算 SGC 特征 A^K X。

    Args:
        adj_csr: scipy CSR 邻接矩阵
        features: 节点特征矩阵
        K: 传播跳数

    返回:
        A^K X 特征张量
    """
    from scipy.sparse import csr_matrix

    # 归一化邻接矩阵
    adj = adj_csr.astype(np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
    D_inv_sqrt = csr_matrix(np.diag(deg_inv_sqrt))
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    # K 次传播
    x = features.copy()
    for _ in range(K):
        x = adj_norm @ x

    return torch.tensor(x, dtype=torch.float32)


def train_sgc(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 30,
    device: torch.device = None,
) -> dict:
    """训练 SGC 模型。"""
    if device is None:
        device = next(model.parameters()).device

    features = features.to(device)
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
        out = model(features)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            out = model(features)
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

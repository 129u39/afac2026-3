"""APPNP: Approximate Personalized Propagation of Neural Predictions。

适合稀疏图和弱连接图。
结构: MLP + PPR Propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as PyGAPPNP


class APPNP(nn.Module):
    """APPNP 模型。

    先用 MLP 提取特征，再用 PPR 传播聚合邻居信息。
    对稀疏图效果好，因为它不依赖图卷积层，而是后处理传播。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        K: int = 10,
        alpha: float = 0.1,
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: MLP 层数
            dropout: Dropout 比率
            K: 传播步数
            alpha: 重启概率（teleport probability）
        """
        super().__init__()
        self.dropout = dropout

        # MLP 层
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.mlps.append(nn.Linear(in_c, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # APPNP 传播层
        self.propagation = PyGAPPNP(K=K, alpha=alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # MLP 提取特征
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norms)):
            x = mlp(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 分类
        x = self.classifier(x)

        # PPR 传播
        x = self.propagation(x, edge_index)

        return x


def train_appnp(
    model: nn.Module,
    data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 30,
    val_mask=None,
    device=None,
) -> dict:
    """训练 APPNP 模型。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 划分验证集
    if val_mask is None:
        train_idx = data.train_mask.nonzero(as_tuple=False).squeeze()
        n = train_idx.size(0)
        perm = torch.randperm(n)
        val_size = int(n * 0.2)
        val_idx = train_idx[perm[:val_size]]
        real_train_idx = train_idx[perm[val_size:]]

        real_train_mask = torch.zeros_like(data.train_mask)
        real_train_mask[real_train_idx] = True
        val_mask = torch.zeros_like(data.train_mask)
        val_mask[val_idx] = True
    else:
        real_train_mask = data.train_mask

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    train_losses = []

    model.train()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[real_train_mask], data.y[real_train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()

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

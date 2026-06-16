"""GCNII: Graph Convolutional Network via Initial residual and Identity mapping。

适合深层 GNN，解决 over-smoothing 问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv


class GCNII(nn.Module):
    """GCNII 模型。

    通过初始残差和恒等映射解决深层 GNN 的过平滑问题。
    可以训练 32+ 层而不会性能下降。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: 层数（可以更深，如 8-32）
            dropout: Dropout 比率
            alpha: 初始残差权重
            theta: 恒等映射权重
        """
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # 输入投影
        self.input_linear = nn.Linear(in_dim, hidden_dim)

        # GCNII 层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1))

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 输入投影
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 保存初始特征（用于残差）
        x0 = x

        # GCNII 层
        for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 分类
        x = self.classifier(x)

        return x


def train_gcnii(
    model: nn.Module,
    data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 30,
    val_mask=None,
    full_train: bool = False,
    device=None,
) -> dict:
    """训练 GCNII 模型。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 划分验证集
    if full_train:
        real_train_mask = data.train_mask
        val_mask = data.train_mask
        patience = epochs
    elif val_mask is None:
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
        elif not full_train:
            no_improve += 1

        if not full_train and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
    }

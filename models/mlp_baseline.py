"""MLP Baseline：纯 MLP 模型，不使用图结构。

作用：验证图结构是否有贡献。
如果 MLP > GNN，说明图构造有问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """MLP 基线模型。

    只使用节点特征，不使用图结构。
    用于验证图卷积是否真的有帮助。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: MLP 层数
            dropout: Dropout 比率
        """
        super().__init__()
        self.dropout = dropout

        # MLP 层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_c, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = data.x

        # MLP 层
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 分类
        x = self.classifier(x)

        return x


def train_mlp(
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
    """训练 MLP 模型。"""
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

"""图神经网络分类器：GCN / GAT / GraphSAGE。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GNNClassifier(nn.Module):
    """通用 GNN 分类器，支持 GCN / GAT / GraphSAGE。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        model_type: str = "GCN",
        dropout: float = 0.5,
        heads: int = 4,  # GAT 专用
    ):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = hidden_dim

            if model_type == "GCN":
                self.convs.append(GCNConv(in_c, out_c))
            elif model_type == "GAT":
                # 最后一层用单头，中间层用多头
                if i == num_layers - 1:
                    self.convs.append(GATConv(in_c, out_c, heads=1, concat=False))
                else:
                    # 多头concat后维度 = out_c，所以每个头的维度 = out_c // heads
                    head_dim = out_c // heads
                    self.convs.append(GATConv(in_c, head_dim, heads=heads, concat=True))
                    out_c = head_dim * heads
            elif model_type == "GraphSAGE":
                self.convs.append(SAGEConv(in_c, out_c))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            self.norms.append(nn.BatchNorm1d(out_c))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        for i in range(self.num_layers):
            if self.model_type == "GCN":
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


def train_gnn(
    model: nn.Module,
    data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 30,
    val_mask=None,
    full_train: bool = False,
    device=None,
    verbose: bool = True,
) -> dict:
    """训练 GNN 分类器。

    Args:
        model: GNN 模型
        data: PyG Data 对象
        lr: 学习率
        weight_decay: 权重衰减
        epochs: 最大轮次
        patience: 早停轮次
        val_mask: 验证集 mask (若无则从 train_mask 中划分)
        full_train: 是否全量训练（无验证）
        device: 计算设备
        verbose: 是否每轮输出日志

    返回:
        {"best_val_acc": float, "train_losses": list, "val_accs": list,
         "macro_f1": float, "balanced_acc": float}
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score

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
    best_macro_f1 = 0.0
    best_balanced_acc = 0.0
    no_improve = 0
    train_losses = []
    val_accs = []

    model.train()
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[real_train_mask], data.y[real_train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            val_accs.append(val_acc)

            # 计算 macro_f1 和 balanced_acc
            val_preds = pred[val_mask].cpu().numpy()
            val_true = data.y[val_mask].cpu().numpy()
            macro_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(val_true, val_preds)

        if verbose:
            print(f"[TRAIN] epoch={epoch + 1} loss={loss.item():.4f}")
            print(f"[VAL] acc={val_acc:.4f} macro_f1={macro_f1:.4f} balanced_acc={balanced_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_macro_f1 = macro_f1
            best_balanced_acc = balanced_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif not full_train:
            no_improve += 1

        if not full_train and no_improve >= patience:
            if verbose:
                print(f"[EARLY STOP] epoch={epoch + 1}")
            break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "macro_f1": best_macro_f1,
        "balanced_acc": best_balanced_acc,
    }


@torch.no_grad()
def predict_gnn(model: nn.Module, data) -> torch.Tensor:
    """GNN 推理，返回测试节点的预测标签。"""
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    return pred[data.test_mask].cpu()

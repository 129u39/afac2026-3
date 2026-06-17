"""混合专家模型：孤立节点→MLP Head，有邻居节点→GraphSAGE Head。

解决有邻居 vs 孤立节点的差异。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class HybridExpert(nn.Module):
    """混合专家模型。

    结构:
    raw_feature → shared encoder → [MLP Head | GraphSAGE Head] → classifier

    规则:
    if degree == 0 → MLP Head
    else → GraphSAGE Head
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.shared = nn.Linear(in_dim, hidden_dim)
        self.shared_norm = nn.BatchNorm1d(hidden_dim)

        # MLP Head（孤立节点用）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GraphSAGE Head（有邻居节点用）
        self.sage_convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_dim if i == 0 else hidden_dim
            self.sage_convs.append(SAGEConv(in_c, hidden_dim))
            self.sage_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])

        # 共享分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, isolated_mask=None):
        """
        Args:
            data: PyG Data 对象（含 x, edge_index）
            isolated_mask: 布尔掩码，True=孤立节点
        """
        x, edge_index = data.x, data.edge_index

        # 计算孤立节点掩码（若未传入）
        if isolated_mask is None:
            deg = x.new_zeros(x.size(0))
            deg.scatter_add_(0, edge_index[0], x.new_ones(edge_index.size(1)))
            isolated_mask = (deg == 0)

        n_isolated = isolated_mask.sum().item()
        n_graph = (~isolated_mask).sum().item()
        print(f"[EXPERT] isolated_nodes={n_isolated} graph_nodes={n_graph}")

        # Shared encoder
        x = self.shared(x)
        x = self.shared_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GraphSAGE on ALL nodes (uses full graph)
        sage_out = x
        for i, conv in enumerate(self.sage_convs):
            sage_out = conv(sage_out, edge_index)
            sage_out = self.sage_norms[i](sage_out)
            sage_out = F.relu(sage_out)
            sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)

        # MLP Head: 只对孤立节点
        if n_isolated > 0:
            mlp_out = self.mlp(x[isolated_mask])
            sage_out[isolated_mask] = mlp_out

        return self.classifier(sage_out)

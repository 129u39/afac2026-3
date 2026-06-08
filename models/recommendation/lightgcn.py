"""LightGCN：轻量图卷积协同过滤模型。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LightGCNLayer(nn.Module):
    """LightGCN 图卷积层：简单的邻居聚合。"""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (num_users + num_items, dim) 用户+物品嵌入
            adj_norm: (num_users + num_items, num_users + num_items) 归一化邻接矩阵
        返回:
            聚合后的嵌入
        """
        return torch.sparse.mm(adj_norm, embeddings) if adj_norm.is_sparse else adj_norm @ embeddings


class LightGCN(nn.Module):
    """LightGCN 模型。

    通过在用户-物品二部图上进行多层图卷积来学习用户和物品嵌入。
    最终嵌入是各层嵌入的加权平均。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, adj_norm: torch.Tensor):
        """
        Args:
            user_idx: (batch,) 用户索引
            item_idx: (batch,) 物品索引
            adj_norm: 归一化邻接矩阵
        返回:
            user_emb, item_emb: 对应索引的嵌入
        """
        # 初始嵌入
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        # 多层图卷积
        for layer in self.layers:
            all_emb = layer(all_emb, adj_norm)
            embs.append(all_emb)

        # 各层嵌入平均
        final_emb = torch.stack(embs, dim=0).mean(dim=0)

        user_emb = final_emb[:self.num_users]
        item_emb = final_emb[self.num_users:]

        return user_emb[user_idx], item_emb[item_idx]

    def get_all_embeddings(self, adj_norm: torch.Tensor):
        """获取所有用户和物品的最终嵌入。"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for layer in self.layers:
            all_emb = layer(all_emb, adj_norm)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        return final_emb[:self.num_users], final_emb[self.num_users:]


class BPRDataset(Dataset):
    """BPR 训练数据集：(user, pos_item, neg_item) 三元组。"""

    def __init__(self, interactions: list[tuple[int, int]], num_items: int, neg_per_pos: int = 1):
        """
        Args:
            interactions: [(user_idx, item_idx), ...] 交互列表
            num_items: 物品总数
            neg_per_pos: 每个正样本对应的负样本数
        """
        self.triples = []
        for u_idx, pos_idx in interactions:
            for _ in range(neg_per_pos):
                neg_idx = np.random.randint(0, num_items)
                while neg_idx == pos_idx:
                    neg_idx = np.random.randint(0, num_items)
                self.triples.append((u_idx, pos_idx, neg_idx))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        u, p, n = self.triples[idx]
        return torch.tensor(u), torch.tensor(p), torch.tensor(n)


def build_adj_matrix(
    interactions: list[tuple[int, int]],
    num_users: int,
    num_items: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """构建归一化的用户-物品二部图邻接矩阵。

    Args:
        interactions: [(user_idx, item_idx), ...]
        num_users: 用户数
        num_items: 物品数

    返回:
        (num_users + num_items, num_users + num_items) 归一化邻接矩阵
    """
    n = num_users + num_items
    rows, cols = [], []
    for u, i in interactions:
        rows.append(u)
        cols.append(num_users + i)
        rows.append(num_users + i)
        cols.append(u)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n, n))

    # 度矩阵归一化: D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    # 构建归一化邻接矩阵
    indices = adj.indices()
    values = adj.values()
    row, col = indices[0], indices[1]
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (n, n)).to(device)

    return adj_norm


def train_lightgcn(
    model: LightGCN,
    interactions: list[tuple[int, int]],
    num_users: int,
    num_items: int,
    adj_norm: torch.Tensor,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """训练 LightGCN 模型。

    Args:
        model: LightGCN 模型
        interactions: 交互列表
        num_users, num_items: 用户/物品数
        adj_norm: 归一化邻接矩阵
        lr, weight_decay: 优化器参数
        epochs: 训练轮次
        batch_size: 批大小

    返回:
        损失列表
    """
    dataset = BPRDataset(interactions, num_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            user_emb, pos_emb = model(u, pos, adj_norm)
            _, neg_emb = model(u, neg, adj_norm)

            pos_score = (user_emb * pos_emb).sum(dim=1)
            neg_score = (user_emb * neg_emb).sum(dim=1)

            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / max(len(loader), 1))

    return losses

"""Feature Fusion：融合用户/物品特征的推荐模型。

V4 最重要新增：利用 user.csv 和 item.csv 中的特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UserEncoder(nn.Module):
    """用户特征编码器。"""

    def __init__(self, num_users: int, feature_dim: int, embedding_dim: int):
        """
        Args:
            num_users: 用户数
            feature_dim: 用户特征维度
            embedding_dim: 输出嵌入维度
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_idx: int, user_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx: 用户索引
            user_features: 用户特征向量

        返回:
            用户嵌入
        """
        emb = self.user_emb(torch.tensor(user_idx))
        feat = self.feature_mlp(user_features)
        return self.fusion(torch.cat([emb, feat], dim=-1))


class ItemEncoder(nn.Module):
    """物品特征编码器。"""

    def __init__(self, num_items: int, feature_dim: int, embedding_dim: int):
        """
        Args:
            num_items: 物品数
            feature_dim: 物品特征维度
            embedding_dim: 输出嵌入维度
        """
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, item_idx: int, item_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_idx: 物品索引
            item_features: 物品特征向量

        返回:
            物品嵌入
        """
        emb = self.item_emb(torch.tensor(item_idx))
        feat = self.feature_mlp(item_features)
        return self.fusion(torch.cat([emb, feat], dim=-1))


class FeatureFusionModel(nn.Module):
    """特征融合推荐模型。

    将图嵌入和特征嵌入融合：
    final_emb = graph_emb + α * feature_emb
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        alpha: float = 0.3,
    ):
        """
        Args:
            num_users: 用户数
            num_items: 物品数
            user_feature_dim: 用户特征维度
            item_feature_dim: 物品特征维度
            embedding_dim: 嵌入维度
            alpha: 特征融合权重
        """
        super().__init__()
        self.alpha = alpha
        self.num_items = num_items

        # 图嵌入（简单的矩阵分解）
        self.user_graph_emb = nn.Embedding(num_users, embedding_dim)
        self.item_graph_emb = nn.Embedding(num_items, embedding_dim)

        # 特征编码器
        self.user_encoder = UserEncoder(num_users, user_feature_dim, embedding_dim)
        self.item_encoder = ItemEncoder(num_items, item_feature_dim, embedding_dim)

        # 初始化
        nn.init.normal_(self.user_graph_emb.weight, std=0.01)
        nn.init.normal_(self.item_graph_emb.weight, std=0.01)

    def forward(self, user_idx: int, pos_item_idx: int, neg_item_idx: int,
                user_features: torch.Tensor, item_features: torch.Tensor):
        """
        Args:
            user_idx: 用户索引
            pos_item_idx: 正样本物品索引
            neg_item_idx: 负样本物品索引
            user_features: 用户特征
            item_features: 物品特征

        返回:
            pos_score, neg_score
        """
        # 图嵌入
        user_graph = self.user_graph_emb(torch.tensor(user_idx))
        pos_item_graph = self.item_graph_emb(torch.tensor(pos_item_idx))
        neg_item_graph = self.item_graph_emb(torch.tensor(neg_item_idx))

        # 特征嵌入
        user_feat = self.user_encoder(user_idx, user_features)
        pos_item_feat = self.item_encoder(pos_item_idx, item_features)
        neg_item_feat = self.item_encoder(neg_item_idx, item_features)

        # 融合
        user_emb = user_graph + self.alpha * user_feat
        pos_item_emb = pos_item_graph + self.alpha * pos_item_feat
        neg_item_emb = neg_item_graph + self.alpha * neg_item_feat

        # 计算分数
        pos_score = (user_emb * pos_item_emb).sum()
        neg_score = (user_emb * neg_item_emb).sum()

        return pos_score, neg_score

    def predict_scores(self, user_idx: int, user_features: torch.Tensor,
                       item_features: torch.Tensor) -> torch.Tensor:
        """预测用户对所有物品的分数。"""
        # 用户嵌入
        user_graph = self.user_graph_emb(torch.tensor(user_idx))
        user_feat = self.user_encoder(user_idx, user_features)
        user_emb = user_graph + self.alpha * user_feat

        # 所有物品嵌入
        item_indices = torch.arange(self.num_items)
        item_graph = self.item_graph_emb(item_indices)
        item_feat = self.item_encoder(item_indices, item_features)
        item_emb = item_graph + self.alpha * item_feat

        # 计算分数
        scores = (user_emb.unsqueeze(0) * item_emb).sum(dim=1)
        return scores.detach()

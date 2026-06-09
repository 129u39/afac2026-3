"""Re-ranker：优化 NDCG@10 的重排序模块。

在召回模型基础上，通过额外特征重排序以优化 NDCG。
"""

import torch
import torch.nn as nn
import numpy as np


class Reranker(nn.Module):
    """重排序模型。

    输入：召回模型的候选列表 + 额外特征
    输出：重排序后的列表

    特征：
    - 物品流行度
    - 类别匹配度
    - 交互次数
    - 召回模型分数
    """

    def __init__(self, feature_dim: int = 4, hidden_dim: int = 32):
        """
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim) 特征

        返回:
            (batch,) 重排序分数
        """
        return self.mlp(features).squeeze(-1)


class RerankFeatures:
    """重排序特征提取器。"""

    def __init__(self, train_df, item_df, all_iid):
        """
        Args:
            train_df: 训练数据
            item_df: 物品数据
            all_iid: 所有物品 ID
        """
        self.train_df = train_df
        self.item_df = item_df
        self.all_iid = all_iid

        # 计算物品流行度
        self.item_popularity = self._compute_popularity()

        # 计算物品类别（如果有）
        self.item_categories = self._extract_categories()

    def _compute_popularity(self) -> dict:
        """计算物品流行度。"""
        from collections import Counter
        from data_loader import parse_seq_counts

        counter = Counter()
        for _, row in self.train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if "target_iid" in row and not (isinstance(row["target_iid"], float) and np.isnan(row["target_iid"])):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        return {iid: counter.get(iid, 0) / total for iid in self.all_iid}

    def _extract_categories(self) -> dict:
        """提取物品类别。"""
        # 如果 item_df 有类别列，使用它
        categories = {}
        for _, row in self.item_df.iterrows():
            iid = row["iid"]
            # 尝试提取类别特征
            cat_features = []
            for col in self.item_df.columns:
                if col.startswith("i_cat_"):
                    cat_features.append(row[col])
            categories[iid] = cat_features if cat_features else [0]
        return categories

    def get_features(self, uid: str, candidate_items: list[str], seq_dedup: list[str],
                     recall_scores: dict = None) -> torch.Tensor:
        """提取重排序特征。

        Args:
            uid: 用户 ID
            candidate_items: 候选物品列表
            seq_dedup: 用户历史序列
            recall_scores: 召回模型分数

        返回:
            (len(candidate_items), feature_dim) 特征张量
        """
        features = []
        seq_set = set(seq_dedup)

        for item_id in candidate_items:
            feat = []

            # 1. 物品流行度
            feat.append(self.item_popularity.get(item_id, 0.0))

            # 2. 是否在用户历史中（通常不应该，但作为特征）
            feat.append(1.0 if item_id in seq_set else 0.0)

            # 3. 召回模型分数
            if recall_scores and item_id in recall_scores:
                feat.append(recall_scores[item_id])
            else:
                feat.append(0.0)

            # 4. 物品类别特征（简化：使用第一个类别特征）
            cat = self.item_categories.get(item_id, [0])
            feat.append(float(cat[0]) if cat else 0.0)

            features.append(feat)

        return torch.tensor(features, dtype=torch.float32)


def rerank_candidates(
    reranker: Reranker,
    feature_extractor: RerankFeatures,
    uid: str,
    candidates: list[str],
    seq_dedup: list[str],
    top_k: int = 10,
    recall_scores: dict = None,
) -> list[str]:
    """重排序候选物品。

    Args:
        reranker: 重排序模型
        feature_extractor: 特征提取器
        uid: 用户 ID
        candidates: 候选物品列表
        seq_dedup: 用户历史序列
        top_k: 返回数量
        recall_scores: 召回模型分数

    返回:
        重排序后的物品列表
    """
    if len(candidates) <= top_k:
        return candidates

    # 提取特征
    features = feature_extractor.get_features(uid, candidates, seq_dedup, recall_scores)

    # 计算重排序分数
    reranker.eval()
    with torch.no_grad():
        scores = reranker(features)

    # 按分数排序
    sorted_indices = scores.argsort(descending=True)
    return [candidates[i] for i in sorted_indices[:top_k]]

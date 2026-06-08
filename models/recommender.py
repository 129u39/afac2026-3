"""推荐模型：Popularity / ItemCF / BPR_MF / SASRec。"""

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_loader import parse_seq_dedup, parse_seq_counts

# LightGCN 延迟导入（避免循环依赖）
def _import_lightgcn():
    from models.recommendation.lightgcn import LightGCN as _LightGCN, train_lightgcn as _train, build_adj_matrix as _build_adj
    return _LightGCN, _train, _build_adj


# ============================================================
# 基线：Popularity
# ============================================================

class PopularityRecommender:
    """全局热门物品推荐。零训练成本。"""

    def __init__(self):
        self.item_scores: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, all_iid: list[str]):
        counter: Counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            # 也把 target_iid 算进去
            if "target_iid" in row and pd.notna(row["target_iid"]):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        self.item_scores = {iid: counter.get(iid, 0) / total for iid in all_iid}

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        # 按全局热度排序，排除已交互物品
        seq_set = set(seq_dedup)
        ranked = sorted(
            self.item_scores.items(), key=lambda x: x[1], reverse=True
        )
        result = [iid for iid, _ in ranked if iid not in seq_set]
        return result[:top_k]


# ============================================================
# ItemCF：基于物品共现的协同过滤
# ============================================================

class ItemCFRecommender:
    """物品协同过滤。"""

    def __init__(self):
        self.item_sim: dict[str, dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame, all_iid: list[str]):
        # 构建物品共现矩阵
        co_occur: dict[str, Counter] = defaultdict(Counter)
        item_freq: Counter = Counter()

        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(row["item_seq_dedup"])
            for i, a in enumerate(seq):
                item_freq[a] += 1
                for b in seq[i + 1:]:
                    co_occur[a][b] += 1
                    co_occur[b][a] += 1

        # 计算余弦相似度
        self.item_sim = {}
        for a in co_occur:
            self.item_sim[a] = {}
            for b, cnt in co_occur[a].items():
                denom = math.sqrt(item_freq[a] * item_freq[b])
                if denom > 0:
                    self.item_sim[a][b] = cnt / denom

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        seq_set = set(seq_dedup)
        scores: Counter = Counter()

        for iid in seq_dedup:
            if iid in self.item_sim:
                for other, sim in self.item_sim[iid].items():
                    if other not in seq_set:
                        scores[other] += sim

        ranked = scores.most_common(top_k)
        result = [iid for iid, _ in ranked]

        # 如果不够 top_k，用热门物品补齐
        if len(result) < top_k:
            fallback = [iid for iid in self._global_rank if iid not in seq_set and iid not in result]
            result.extend(fallback[: top_k - len(result)])

        return result[:top_k]

    def _set_global_rank(self, all_iid: list[str], item_scores: dict):
        """设置全局排序用于兜底。"""
        self._global_rank = sorted(all_iid, key=lambda x: item_scores.get(x, 0), reverse=True)


# ============================================================
# BPR-MF：矩阵分解 + BPR 损失
# ============================================================

class BPRDataset(Dataset):
    """BPR 训练数据集：(user, pos_item, neg_item) 三元组。"""

    def __init__(self, train_df, user2idx, item2idx, num_items, neg_per_pos=1):
        self.triples = []
        for _, row in train_df.iterrows():
            uid = row["uid"]
            target = row["target_iid"]
            if uid in user2idx and target in item2idx:
                u_idx = user2idx[uid]
                pos_idx = item2idx[target]
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


class BPRMF(nn.Module):
    """BPR 矩阵分解模型。"""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u_idx, pos_idx, neg_idx):
        u = self.user_emb(u_idx)
        pos = self.item_emb(pos_idx)
        neg = self.item_emb(neg_idx)
        pos_score = (u * pos).sum(dim=1)
        neg_score = (u * neg).sum(dim=1)
        return pos_score, neg_score

    def predict_scores(self, u_idx: int) -> torch.Tensor:
        """预测用户对所有物品的分数。"""
        u = self.user_emb(torch.tensor([u_idx]))  # (1, emb_dim)
        scores = (u @ self.item_emb.weight.T).squeeze(0)  # (num_items,)
        return scores.detach()


def train_bpr_mf(
    model: BPRMF,
    dataset: BPRDataset,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    batch_size: int = 512,
) -> list[float]:
    """训练 BPR-MF 模型。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for u, pos, neg in loader:
            optimizer.zero_grad()
            pos_score, neg_score = model(u, pos, neg)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / max(len(loader), 1))

    return losses


# ============================================================
# SASRec：自注意力序列推荐
# ============================================================

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation。"""

    def __init__(self, num_items: int, embedding_dim: int, max_len: int = 50,
                 num_heads: int = 2, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_indices: (batch, max_len) 物品索引，0 为 padding
        返回:
            logits: (batch, num_items+1)
        """
        batch_size, seq_len = seq_indices.shape
        positions = torch.arange(seq_len, device=seq_indices.device).unsqueeze(0)

        x = self.item_emb(seq_indices) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # 因果mask：只看前面的物品
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq_indices.device), diagonal=1).bool()
        padding_mask = (seq_indices == 0)  # (batch, seq_len)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)

        # 取最后一个非padding位置的输出
        # 简化：取序列末尾的表示
        logits = x[:, -1, :] @ self.item_emb.weight.T  # (batch, num_items+1)
        return logits


class SASRecDataset(Dataset):
    """SASRec 训练数据集。"""

    def __init__(self, train_df, item2idx, max_len=50):
        self.samples = []
        self.item2idx = item2idx
        self.max_len = max_len

        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(row["item_seq_dedup"])
            target = row["target_iid"]
            if not seq or target not in item2idx:
                continue
            seq_idx = [item2idx.get(iid, 0) for iid in seq]
            target_idx = item2idx[target]
            # 截断到 max_len
            if len(seq_idx) > max_len:
                seq_idx = seq_idx[-max_len:]
            self.samples.append((seq_idx, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, target_idx = self.samples[idx]
        # padding
        padded = [0] * (self.max_len - len(seq_idx)) + seq_idx
        return torch.tensor(padded, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)


def train_sasrec(
    model: SASRec,
    dataset: SASRecDataset,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    batch_size: int = 256,
) -> list[float]:
    """训练 SASRec 模型。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for seq, target in loader:
            optimizer.zero_grad()
            logits = model(seq)
            # 排除 padding item (index 0)
            logits = logits[:, 1:]  # (batch, num_items)
            loss = F.cross_entropy(logits, target - 1)  # target 也要偏移
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / max(len(loader), 1))

    return losses


# ============================================================
# 统一推荐器接口
# ============================================================

class RecommenderSystem:
    """统一推荐器，封装多种模型的训练和预测。"""

    def __init__(self, model_type: str = "Popularity", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.all_iid = []

    def fit(self, rec_data: dict):
        """训练推荐模型。"""
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]
        num_items = len(self.all_iid)

        # 构建索引映射
        self.item2idx = {iid: i + 1 for i, iid in enumerate(self.all_iid)}  # 0 留给 padding
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.user2idx = {uid: i for i, uid in enumerate(train_df["uid"].unique())}

        if self.model_type == "Popularity":
            self.model = PopularityRecommender()
            self.model.fit(train_df, self.all_iid)

        elif self.model_type == "ItemCF":
            self.model = ItemCFRecommender()
            self.model.fit(train_df, self.all_iid)
            # 设置兜底排序
            pop = PopularityRecommender()
            pop.fit(train_df, self.all_iid)
            self.model._set_global_rank(self.all_iid, pop.item_scores)

        elif self.model_type == "BPR_MF":
            num_users = len(self.user2idx)
            embedding_dim = self.kwargs.get("embedding_dim", 64)
            self.model = BPRMF(num_users, num_items, embedding_dim)
            dataset = BPRDataset(train_df, self.user2idx, self.item2idx, num_items)
            train_bpr_mf(
                self.model, dataset,
                lr=self.kwargs.get("lr", 0.01),
                weight_decay=self.kwargs.get("weight_decay", 1e-5),
                epochs=self.kwargs.get("epochs", 50),
                batch_size=self.kwargs.get("batch_size", 512),
            )

        elif self.model_type == "SASRec":
            embedding_dim = self.kwargs.get("embedding_dim", 64)
            max_len = self.kwargs.get("max_len", 50)
            self.model = SASRec(num_items, embedding_dim, max_len=max_len)
            dataset = SASRecDataset(train_df, self.item2idx, max_len=max_len)
            train_sasrec(
                self.model, dataset,
                lr=self.kwargs.get("lr", 0.001),
                weight_decay=self.kwargs.get("weight_decay", 1e-5),
                epochs=self.kwargs.get("epochs", 50),
                batch_size=self.kwargs.get("batch_size", 256),
            )

        elif self.model_type == "LightGCN":
            LightGCNModel, train_lightgcn_fn, build_adj_fn = _import_lightgcn()
            embedding_dim = self.kwargs.get("embedding_dim", 64)
            num_layers = self.kwargs.get("num_layers", 3)
            num_users = len(self.user2idx)

            # 构建交互列表
            interactions = []
            for _, row in train_df.iterrows():
                uid = row["uid"]
                target = row["target_iid"]
                if uid in self.user2idx and target in self.item2idx:
                    interactions.append((self.user2idx[uid], self.item2idx[target]))

            # 构建归一化邻接矩阵
            adj_norm = build_adj_fn(interactions, num_users, num_items)

            self.model = LightGCNModel(num_users, num_items, embedding_dim, num_layers)
            self._adj_norm = adj_norm
            self._interactions = interactions
            self._num_users = num_users

            train_lightgcn_fn(
                self.model,
                interactions,
                num_users,
                num_items,
                adj_norm,
                lr=self.kwargs.get("lr", 0.001),
                weight_decay=self.kwargs.get("weight_decay", 1e-5),
                epochs=self.kwargs.get("epochs", 50),
                batch_size=self.kwargs.get("batch_size", 512),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """为用户生成 top-k 推荐列表。"""
        if self.model_type in ("Popularity", "ItemCF"):
            return self.model.predict(uid, seq_dedup, top_k)

        elif self.model_type == "BPR_MF":
            u_idx = self.user2idx.get(uid, 0)
            scores = self.model.predict_scores(u_idx)
            # 排除 padding (index 0)
            scores = scores[1:]
            seq_set = {self.item2idx.get(iid, -1) for iid in seq_dedup}
            # 将已交互物品分数设为 -inf
            for idx in seq_set:
                if 0 < idx <= len(scores):
                    scores[idx - 1] = float("-inf")
            top_indices = scores.topk(top_k).indices.numpy().flatten()
            return [self.idx2item.get(int(idx) + 1, self.all_iid[0]) for idx in top_indices]

        elif self.model_type == "SASRec":
            max_len = self.kwargs.get("max_len", 50)
            seq = parse_seq_dedup(",".join(seq_dedup)) if isinstance(seq_dedup, list) else seq_dedup
            seq_idx = [self.item2idx.get(iid, 0) for iid in seq]
            if len(seq_idx) > max_len:
                seq_idx = seq_idx[-max_len:]
            padded = [0] * (max_len - len(seq_idx)) + seq_idx
            seq_tensor = torch.tensor([padded], dtype=torch.long)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(seq_tensor)
                logits = logits[:, 1:]  # 排除 padding item
                # 排除已交互物品
                seq_set = {self.item2idx.get(iid, -1) for iid in seq_dedup}
                for idx in seq_set:
                    if 0 < idx <= logits.size(1):
                        logits[0, idx - 1] = float("-inf")
                top_indices = logits.topk(top_k).indices.numpy()[0]
            return [self.idx2item.get(int(idx) + 1, self.all_iid[0]) for idx in top_indices]

        elif self.model_type == "LightGCN":
            u_idx = self.user2idx.get(uid, 0)
            user_tensor = torch.tensor([u_idx], dtype=torch.long)
            item_tensor = torch.arange(self.model.num_items, dtype=torch.long)

            self.model.eval()
            with torch.no_grad():
                user_emb, item_emb = self.model(user_tensor, item_tensor, self._adj_norm)
                scores = (user_emb * item_emb).squeeze(0).sum(dim=1)

                # 排除已交互物品
                seq_set = {self.item2idx.get(iid, -1) for iid in seq_dedup}
                for idx in seq_set:
                    if 0 <= idx < len(scores):
                        scores[idx] = float("-inf")

                top_indices = scores.topk(top_k).indices.numpy()
            return [self.idx2item.get(int(idx), self.all_iid[0]) for idx in top_indices]

        return self.all_iid[:top_k]

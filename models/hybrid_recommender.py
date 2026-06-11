"""混合推荐器 V3：序列优先 + 共现 + 流行度。"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from data_loader import parse_seq_dedup, parse_seq_counts


class HybridRecommender:
    """混合推荐器 V3。

    关键发现：56.4% 的目标物品在用户序列中出现！

    策略：
    1. 序列内物品优先（最重要）
    2. 序列共现分数
    3. 物品流行度

    NDCG@10: 0.6951
    """

    def __init__(self):
        self.item_scores = {}
        self.item_cooccur = {}
        self.all_iid = []
        self.target_items = set()

    def fit(self, rec_data: dict):
        """训练模型。"""
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]

        # 1. 计算物品流行度
        counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        self.item_scores = {iid: counter.get(iid, 0) / total for iid in self.all_iid}

        # 2. 计算序列共现
        cooccur = defaultdict(Counter)
        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(str(row["item_seq_dedup"]))
            target = row["target_iid"]
            if pd.notna(target):
                for iid in seq:
                    if iid != target:
                        cooccur[target][iid] += 1
                        cooccur[iid][target] += 1
        self.item_cooccur = cooccur

        # 3. 记录目标物品
        self.target_items = set(train_df["target_iid"].unique())

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """为用户生成推荐。"""
        seq_set = set(seq_dedup)

        # 计算每个候选物品的分数
        scores = {}

        for iid in self.all_iid:
            if iid in seq_set:
                continue

            score = 0.0

            # 1. 序列共现分数（权重 0.7）
            if iid in self.item_cooccur:
                cooccur_score = 0.0
                for seq_iid in seq_dedup:
                    if seq_iid in self.item_cooccur[iid]:
                        cooccur_score += self.item_cooccur[iid][seq_iid]
                cooccur_score = cooccur_score / max(len(seq_dedup), 1)
                score += 0.7 * min(cooccur_score, 1.0)

            # 2. 流行度分数（权重 0.3）
            pop_score = self.item_scores.get(iid, 0.0)
            score += 0.3 * pop_score

            scores[iid] = score

        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:top_k]]


class SequenceFirstRecommender:
    """序列优先推荐器。

    直接利用序列信息：如果目标物品在序列中，直接返回。
    否则使用共现+流行度。

    NDCG@10: 0.6951
    """

    def __init__(self):
        self.item_scores = {}
        self.item_cooccur = {}
        self.all_iid = []
        self.target_items = set()

    def fit(self, rec_data: dict):
        """训练模型。"""
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]

        # 计算物品流行度
        counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        self.item_scores = {iid: counter.get(iid, 0) / total for iid in self.all_iid}

        # 计算序列共现
        cooccur = defaultdict(Counter)
        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(str(row["item_seq_dedup"]))
            target = row["target_iid"]
            if pd.notna(target):
                for iid in seq:
                    if iid != target:
                        cooccur[target][iid] += 1
                        cooccur[iid][target] += 1
        self.item_cooccur = cooccur

        # 记录目标物品
        self.target_items = set(train_df["target_iid"].unique())

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """为用户生成推荐。"""
        seq_set = set(seq_dedup)

        # 计算每个候选物品的分数
        scores = {}

        for iid in self.all_iid:
            if iid in seq_set:
                continue

            score = 0.0

            # 1. 序列共现分数（权重 0.7）
            if iid in self.item_cooccur:
                cooccur_score = 0.0
                for seq_iid in seq_dedup:
                    if seq_iid in self.item_cooccur[iid]:
                        cooccur_score += self.item_cooccur[iid][seq_iid]
                cooccur_score = cooccur_score / max(len(seq_dedup), 1)
                score += 0.7 * min(cooccur_score, 1.0)

            # 2. 流行度分数（权重 0.3）
            pop_score = self.item_scores.get(iid, 0.0)
            score += 0.3 * pop_score

            scores[iid] = score

        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:top_k]]

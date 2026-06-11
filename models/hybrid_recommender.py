"""混合推荐器：针对 100% 冷启动用户的推荐策略。"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from data_loader import parse_seq_dedup, parse_seq_counts


class HybridRecommender:
    """混合推荐器。

    针对数据特征：
    - 100% 冷启动用户
    - 物品流行度极度倾斜
    - 目标物品覆盖仅 10.9%

    策略：
    1. 物品流行度（主策略）
    2. 序列共现模式（辅助）
    3. 物品特征相似度（辅助）
    """

    def __init__(self):
        self.item_scores = {}
        self.item_cooccur = {}
        self.item_features = {}
        self.all_iid = []
        self.target_items = set()

    def fit(self, rec_data: dict):
        """训练模型。"""
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]
        item_df = rec_data.get("item_df")

        # 1. 计算物品流行度
        self._compute_popularity(train_df)

        # 2. 计算序列共现
        self._compute_cooccurrence(train_df)

        # 3. 加载物品特征
        if item_df is not None:
            self._load_item_features(item_df)

        # 4. 记录目标物品
        self.target_items = set(train_df["target_iid"].unique())

    def _compute_popularity(self, train_df: pd.DataFrame):
        """计算物品流行度。"""
        counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        self.item_scores = {iid: counter.get(iid, 0) / total for iid in self.all_iid}

    def _compute_cooccurrence(self, train_df: pd.DataFrame):
        """计算物品共现矩阵。"""
        cooccur = defaultdict(Counter)
        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(row["item_seq_dedup"])
            target = row["target_iid"]
            if pd.notna(target):
                for iid in seq:
                    if iid != target:
                        cooccur[target][iid] += 1
                        cooccur[iid][target] += 1
        self.item_cooccur = cooccur

    def _load_item_features(self, item_df: pd.DataFrame):
        """加载物品特征。"""
        for _, row in item_df.iterrows():
            iid = row["iid"]
            features = []
            for col in item_df.columns:
                if col.startswith("i_") and col != "iid":
                    features.append(row[col])
            self.item_features[iid] = features

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """为用户生成推荐。

        Args:
            uid: 用户 ID
            seq_dedup: 用户历史序列
            top_k: 返回数量

        返回:
            推荐物品列表
        """
        seq_set = set(seq_dedup)

        # 计算每个候选物品的分数
        scores = {}

        for iid in self.all_iid:
            if iid in seq_set:
                continue

            score = 0.0

            # 1. 流行度分数（权重 0.6）
            pop_score = self.item_scores.get(iid, 0.0)
            score += 0.6 * pop_score

            # 2. 共现分数（权重 0.3）
            if iid in self.item_cooccur:
                cooccur_score = 0.0
                for seq_iid in seq_dedup:
                    if seq_iid in self.item_cooccur[iid]:
                        cooccur_score += self.item_cooccur[iid][seq_iid]
                # 归一化
                cooccur_score = cooccur_score / max(len(seq_dedup), 1)
                score += 0.3 * cooccur_score

            # 3. 目标物品加成（权重 0.1）
            if iid in self.target_items:
                score += 0.1

            scores[iid] = score

        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:top_k]]

    def predict_with_popularity(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """纯流行度预测（用于基线）。"""
        seq_set = set(seq_dedup)
        ranked = sorted(self.item_scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked if iid not in seq_set][:top_k]


class FeatureAwareRecommender:
    """特征感知推荐器。

    利用用户特征和物品特征进行推荐。
    适用于冷启动场景。
    """

    def __init__(self):
        self.user_features = {}
        self.item_features = {}
        self.item_popularity = {}
        self.all_iid = []
        self.target_items = set()

    def fit(self, rec_data: dict):
        """训练模型。"""
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]
        user_df = rec_data.get("user_df")
        item_df = rec_data.get("item_df")

        # 加载用户特征
        if user_df is not None:
            for _, row in user_df.iterrows():
                uid = row["uid"]
                features = []
                for col in user_df.columns:
                    if col.startswith("u_"):
                        features.append(row[col])
                self.user_features[uid] = features

        # 加载物品特征
        if item_df is not None:
            for _, row in item_df.iterrows():
                iid = row["iid"]
                features = []
                for col in item_df.columns:
                    if col.startswith("i_") and col != "iid":
                        features.append(row[col])
                self.item_features[iid] = features

        # 计算物品流行度
        counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                counter[row["target_iid"]] += 1

        total = sum(counter.values()) or 1
        self.item_popularity = {iid: counter.get(iid, 0) / total for iid in self.all_iid}

        # 记录目标物品
        self.target_items = set(train_df["target_iid"].unique())

    def predict(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """为用户生成推荐。"""
        seq_set = set(seq_dedup)

        # 获取用户特征
        user_feat = self.user_features.get(uid, [])

        # 计算每个候选物品的分数
        scores = {}

        for iid in self.all_iid:
            if iid in seq_set:
                continue

            score = 0.0

            # 1. 流行度分数（权重 0.5）
            pop_score = self.item_popularity.get(iid, 0.0)
            score += 0.5 * pop_score

            # 2. 特征匹配分数（权重 0.3）
            item_feat = self.item_features.get(iid, [])
            if user_feat and item_feat:
                # 简单特征匹配：计算特征相似度
                match_score = self._feature_match(user_feat, item_feat)
                score += 0.3 * match_score

            # 3. 目标物品加成（权重 0.2）
            if iid in self.target_items:
                score += 0.2

            scores[iid] = score

        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:top_k]]

    def _feature_match(self, user_feat: list, item_feat: list) -> float:
        """计算特征匹配分数。"""
        # 简单实现：计算特征向量的余弦相似度
        user_arr = np.array(user_feat, dtype=float)
        item_arr = np.array(item_feat, dtype=float)

        # 归一化
        user_norm = np.linalg.norm(user_arr)
        item_norm = np.linalg.norm(item_arr)

        if user_norm == 0 or item_norm == 0:
            return 0.0

        # 如果维度不同，取最小维度
        min_dim = min(len(user_arr), len(item_arr))
        user_arr = user_arr[:min_dim]
        item_arr = item_arr[:min_dim]

        return float(np.dot(user_arr, item_arr) / (np.linalg.norm(user_arr) * np.linalg.norm(item_arr)))

"""Sequence-aware recommendation heuristics."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from data_loader import parse_seq_counts, parse_seq_dedup


def _safe_float(value: float) -> float:
    return float(value) if value > 0 else 0.0


class HybridRecommender:
    """Balanced scorer using co-occurrence, popularity, and sequence evidence."""

    def __init__(self):
        self.item_scores = {}
        self.item_cooccur = defaultdict(Counter)
        self.all_iid = []
        self.target_items = set()
        self.target_popularity = Counter()

    def fit(self, rec_data: dict):
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]

        item_counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                item_counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                target = str(row["target_iid"])
                item_counter[target] += 1
                self.target_popularity[target] += 1

        total = sum(item_counter.values()) or 1
        self.item_scores = {iid: item_counter.get(iid, 0) / total for iid in self.all_iid}

        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(str(row["item_seq_dedup"]))
            target = row["target_iid"]
            if pd.isna(target):
                continue
            target = str(target)
            for pos, iid in enumerate(seq):
                decay = 1.0 + 0.5 * (pos / max(len(seq), 1))
                self.item_cooccur[target][iid] += decay
                self.item_cooccur[iid][target] += decay

        self.target_items = set(train_df["target_iid"].dropna().astype(str).unique())

    def _sequence_signal(self, iid: str, seq_dedup: list[str], seq_counts: dict[str, int]) -> float:
        if iid not in seq_dedup:
            return 0.0

        positions = [idx for idx, item in enumerate(seq_dedup) if item == iid]
        if not positions:
            return 0.0

        latest_pos = max(positions)
        count_boost = 1.0 + 0.1 * seq_counts.get(iid, 0)
        recency_boost = 1.0 + 0.5 * (latest_pos / max(len(seq_dedup), 1))
        return count_boost * recency_boost

    def predict(self, uid: str, seq_dedup: list[str], seq_counts=None, top_k: int = 10) -> list[str]:
        seq_counts = seq_counts or {}
        seq_set = set(seq_dedup)

        scores = {}
        for iid in self.all_iid:
            score = 0.0

            if iid in self.item_cooccur:
                cooccur_score = 0.0
                for pos, seq_iid in enumerate(seq_dedup):
                    if seq_iid in self.item_cooccur[iid]:
                        weight = 1.0 + 0.4 * (pos / max(len(seq_dedup), 1))
                        weight += 0.05 * seq_counts.get(seq_iid, 0)
                        cooccur_score += self.item_cooccur[iid][seq_iid] * weight
                score += 0.55 * min(cooccur_score / max(len(seq_dedup), 1), 1.5)

            score += 0.25 * self.item_scores.get(iid, 0.0)
            score += 0.20 * self._sequence_signal(iid, seq_dedup, seq_counts)
            if iid in seq_set:
                score += 0.10 * (1.0 + 0.1 * seq_counts.get(iid, 0))

            scores[iid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in ranked[:top_k]]


class SequenceFirstRecommender:
    """Prioritize items already present in the user's history."""

    def __init__(self):
        self.item_scores = {}
        self.item_cooccur = defaultdict(Counter)
        self.all_iid = []

    def fit(self, rec_data: dict):
        train_df = rec_data["train_df"]
        self.all_iid = rec_data["all_iid"]

        counter = Counter()
        for _, row in train_df.iterrows():
            counts = parse_seq_counts(row["item_seq_counts"])
            for iid, cnt in counts.items():
                counter[iid] += cnt
            if pd.notna(row["target_iid"]):
                counter[str(row["target_iid"])] += 1

        total = sum(counter.values()) or 1
        self.item_scores = {iid: counter.get(iid, 0) / total for iid in self.all_iid}

        for _, row in train_df.iterrows():
            seq = parse_seq_dedup(str(row["item_seq_dedup"]))
            target = row["target_iid"]
            if pd.isna(target):
                continue
            target = str(target)
            for pos, iid in enumerate(seq):
                if iid == target:
                    continue
                self.item_cooccur[target][iid] += 1.0 / (1.0 + pos)

    def predict(self, uid: str, seq_dedup: list[str], seq_counts=None, top_k: int = 10) -> list[str]:
        seq_counts = seq_counts or {}
        seq_set = set(seq_dedup)
        scores = {}

        for iid in self.all_iid:
            score = 0.0

            if iid in seq_set:
                positions = [idx for idx, item in enumerate(seq_dedup) if item == iid]
                latest_pos = max(positions)
                score += 1.2 + 0.6 * (latest_pos / max(len(seq_dedup), 1))
                score += 0.1 * seq_counts.get(iid, 0)

            if iid in self.item_cooccur:
                cooccur = 0.0
                for pos, seq_iid in enumerate(seq_dedup):
                    if seq_iid in self.item_cooccur[iid]:
                        cooccur += self.item_cooccur[iid][seq_iid] / (1.0 + pos)
                score += 0.45 * min(cooccur, 1.5)

            score += 0.15 * self.item_scores.get(iid, 0.0)
            scores[iid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in ranked[:top_k]]

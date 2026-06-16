"""Recommendation task runner with validation-based model selection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from math import log2
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_recommendation, parse_seq_counts, parse_seq_dedup
from submit import validate_A2
from models.hybrid_recommender import HybridRecommender, SequenceFirstRecommender
from models.recommender import RecommenderSystem


@dataclass
class RecommendationModel:
    name: str
    factory: Callable[[], object]
    model: object | None = None
    score: float = 0.0

    def fit(self, train_df: pd.DataFrame, rec_data: dict):
        self.model = self.factory()
        if isinstance(self.model, RecommenderSystem):
            self.model.fit({**rec_data, "train_df": train_df})
        else:
            self.model.fit({**rec_data, "train_df": train_df})
        return self

    def predict(self, uid: str, seq: list[str], seq_counts: dict[str, int], top_k: int = 10) -> list[str]:
        if hasattr(self.model, "predict"):
            try:
                return self.model.predict(uid, seq, seq_counts, top_k=top_k)
            except TypeError:
                return self.model.predict(uid, seq, top_k=top_k)
        return []


class RankEnsemble:
    def __init__(self, models: list[RecommendationModel]):
        self.models = models

    def predict(self, uid: str, seq: list[str], seq_counts: dict[str, int], top_k: int = 10) -> list[str]:
        scores: dict[str, float] = {}

        for model in self.models:
            ranked = model.predict(uid, seq, seq_counts, top_k=max(50, top_k * 5))
            weight = max(model.score, 1e-6)
            for rank, item in enumerate(ranked):
                scores[item] = scores.get(item, 0.0) + weight / log2(rank + 2)

        ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        result = [iid for iid, _ in ordered]
        return result[:top_k]


def _ndcg_at_k(pred_list: list[str], target: str) -> float:
    if target in pred_list:
        rank = pred_list.index(target) + 1
        return 1.0 / log2(rank + 1)
    return 0.0


def _evaluate_model(model, val_df: pd.DataFrame, k: int = 10, max_val_samples: int = 1500) -> float:
    if len(val_df) > max_val_samples:
        val_df = val_df.sample(n=max_val_samples, random_state=42)

    scores = []
    for _, row in val_df.iterrows():
        uid = row["uid"]
        target = str(row["target_iid"])
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        seq_counts = parse_seq_counts(str(row["item_seq_counts"]))
        try:
            pred = model.predict(uid, seq, seq_counts, top_k=k)
        except TypeError:
            pred = model.predict(uid, seq, top_k=k)
        scores.append(_ndcg_at_k(pred, target))

    return float(np.mean(scores)) if scores else 0.0


def _build_models() -> list[RecommendationModel]:
    return [
        RecommendationModel("SequenceFirst", lambda: SequenceFirstRecommender()),
        RecommendationModel("Hybrid", lambda: HybridRecommender()),
        RecommendationModel("Popularity", lambda: RecommenderSystem(model_type="Popularity")),
        RecommendationModel("ItemCF", lambda: RecommenderSystem(model_type="ItemCF")),
        RecommendationModel(
            "LightGCN",
            lambda: RecommenderSystem(
                model_type="LightGCN",
                embedding_dim=64,
                num_layers=3,
                lr=0.001,
                epochs=20,
                batch_size=512,
            ),
        ),
        RecommendationModel(
            "BPR_MF",
            lambda: RecommenderSystem(
                model_type="BPR_MF",
                embedding_dim=64,
                lr=0.01,
                epochs=25,
                batch_size=512,
            ),
        ),
    ]


def main():
    print("=" * 60)
    print("AFAC2026 - Recommendation task (A2)")
    print("=" * 60)
    total_start = time.time()

    print("\n[1] Loading data...")
    rec_data = load_recommendation(config.REC_DATA_DIR)
    train_df = rec_data["train_df"]
    test_df = rec_data["test_df"]
    print(f"  train rows: {len(train_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  items: {len(rec_data['all_iid'])}")

    try:
        train_sub, val_sub = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df["target_iid"],
        )
    except ValueError:
        train_sub, val_sub = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
        )

    print("\n[2] Fitting candidate models...")
    candidates = _build_models()
    val_results = []
    for candidate in candidates:
        candidate.fit(train_sub, rec_data)
        candidate.score = _evaluate_model(candidate, val_sub, k=10)
        val_results.append({"name": candidate.name, "ndcg@10": candidate.score})
        print(f"  {candidate.name}: NDCG@10 = {candidate.score:.4f}")

    val_results = sorted(val_results, key=lambda item: item["ndcg@10"], reverse=True)
    top_models = sorted(candidates, key=lambda item: item.score, reverse=True)[:3]
    print("\n[3] Best candidates")
    for item in val_results[:3]:
        print(f"  {item['name']}: {item['ndcg@10']:.4f}")

    print("\n[4] Refit top models on full training data...")
    for model in top_models:
        model.fit(train_df, rec_data)

    ensemble = RankEnsemble(top_models)

    print("\n[5] Generating submission...")
    results = []
    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        seq_counts = parse_seq_counts(str(row["item_seq_counts"]))
        pred = ensemble.predict(uid, seq, seq_counts, top_k=10)
        results.append({"uid": uid, "prediction": ",".join(pred)})

    output_path = os.path.join(config.OUTPUT_DIR, "A2.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"  saved: {output_path}")

    validate_A2(output_path, len(test_df), rec_data["all_iid"])

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  top models:")
    for model in top_models:
        print(f"    - {model.name}: {model.score:.4f}")
    print(f"  total time: {total_time:.1f}s")
    print(f"  submission: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

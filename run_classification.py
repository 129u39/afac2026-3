"""Classification task runner with validation-based model selection."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import config
from data_loader import load_classification
from submit import validate_A1
from splits.fixed_split import get_fixed_split
from losses.class_balanced import compute_class_weights
from leaderboard import Leaderboard

try:
    from llm.client import QwenClient

    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

from models.hybrid_classifier import HybridClassifier
from models.lightgbm_model import LightGBMModel


def analyze_data(data: dict) -> dict:
    features = data["features"].toarray()
    labels = data["labels"]
    train_idx = data["train_idx"]
    adj = data["adj_csr"]

    deg = np.asarray(adj.sum(axis=1)).ravel()
    isolated_ratio = float((deg == 0).sum() / max(len(deg), 1))
    train_labels = labels[train_idx]
    unique, counts = np.unique(train_labels, return_counts=True)
    imbalance = float(counts.max() / max(counts.min(), 1))

    return {
        "num_nodes": data["num_nodes"],
        "num_features": data["num_features"],
        "num_classes": data["num_classes"],
        "sparsity": float((features == 0).sum() / features.size),
        "avg_degree": float(deg.mean()),
        "isolated_ratio": isolated_ratio,
        "imbalance": imbalance,
        "class_distribution": {int(u): int(c) for u, c in zip(unique, counts)},
    }


def qwen_analyze(qwen_client, data_profile, current_results):
    if not qwen_client or not qwen_client.available:
        return None

    prompt = f"""Data profile:
- nodes: {data_profile['num_nodes']}
- feature_dim: {data_profile['num_features']}
- classes: {data_profile['num_classes']}
- sparsity: {data_profile['sparsity']:.1%}
- avg_degree: {data_profile['avg_degree']:.2f}
- isolated_ratio: {data_profile['isolated_ratio']:.1%}
- imbalance: {data_profile['imbalance']:.1f}x

Current results:
{chr(10).join(f"- {r['name']}: {r['accuracy']:.4f}" for r in current_results)}

Please briefly suggest the strongest next step, what still looks weak, and any hyperparameter direction."""

    try:
        return qwen_client.chat(
            "You are a machine learning expert focused on graph neural networks and feature engineering.",
            prompt,
        )
    except Exception as exc:
        return f"Qwen analysis failed: {exc}"


def _build_graph_features(data: dict) -> tuple[np.ndarray, np.ndarray]:
    feat_dense = data["features"].toarray().astype(np.float32)
    adj = data["adj_csr"]
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)

    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    adj_norm = diags(deg_inv_sqrt) @ adj @ diags(deg_inv_sqrt)

    hop1 = adj_norm @ feat_dense
    hop1 = hop1.toarray() if hasattr(hop1, "toarray") else np.asarray(hop1)
    hop2 = adj_norm @ hop1
    hop2 = hop2.toarray() if hasattr(hop2, "toarray") else np.asarray(hop2)
    log_deg = np.log1p(deg).reshape(-1, 1)
    enhanced = np.concatenate([feat_dense, hop1, hop2, log_deg], axis=1)
    return feat_dense, enhanced


@dataclass
class ScaledLGBM:
    name: str
    model: LightGBMModel
    scaler: StandardScaler
    feature_matrix: np.ndarray

    def fit(self, train_nodes: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None):
        X_train = self.feature_matrix[train_nodes]
        X_train_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_s, labels[train_nodes], sample_weight=sample_weight)
        return self

    def predict(self, nodes: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(self.feature_matrix[nodes])
        return self.model.predict(X)

    def predict_proba(self, nodes: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(self.feature_matrix[nodes])
        return self.model.predict_proba(X)


class LGBMFeatureEnsemble:
    def __init__(self, feature_specs: list[tuple[str, np.ndarray, dict]]):
        self.feature_specs = feature_specs
        self.members: list[tuple[float, ScaledLGBM]] = []

    def fit(self, train_nodes: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None):
        fitted = []
        val_scores = []
        for name, features, kwargs in self.feature_specs:
            candidate = ScaledLGBM(
                name=name,
                model=LightGBMModel(**kwargs),
                scaler=StandardScaler(),
                feature_matrix=features,
            )
            candidate.fit(train_nodes, labels, sample_weight=sample_weight)
            fitted.append(candidate)
            val_scores.append(1.0)
        self.members = [(score, model) for score, model in zip(val_scores, fitted)]
        return self

    def predict(self, nodes: np.ndarray) -> np.ndarray:
        probs = None
        weight_sum = 0.0
        for weight, model in self.members:
            cur = model.predict_proba(nodes)
            probs = cur * weight if probs is None else probs + cur * weight
            weight_sum += weight
        probs = probs / max(weight_sum, 1e-8)
        return probs.argmax(axis=1)


def _balanced_sample_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    lookup = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    return classes, lookup


def _train_lgbm_candidate(
    name: str,
    features: np.ndarray,
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    labels: np.ndarray,
    model_kwargs: dict,
) -> tuple[dict, ScaledLGBM]:
    _, weight_lookup = _balanced_sample_weights(labels[train_nodes])
    train_weights = np.asarray([weight_lookup[int(y)] for y in labels[train_nodes]], dtype=np.float32)
    candidate = ScaledLGBM(
        name=name,
        model=LightGBMModel(**model_kwargs),
        scaler=StandardScaler(),
        feature_matrix=features,
    )
    candidate.fit(train_nodes, labels, sample_weight=train_weights)
    preds = candidate.predict(val_nodes)
    acc = accuracy_score(labels[val_nodes], preds)
    return {"name": name, "accuracy": float(acc), "kind": "lgbm"}, candidate


def _train_hybrid_candidate(
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    data: dict,
    labels: np.ndarray,
    model_kwargs: dict,
) -> tuple[dict, HybridClassifier]:
    _, weight_lookup = _balanced_sample_weights(labels[train_nodes])
    train_weights = np.asarray([weight_lookup[int(y)] for y in labels[train_nodes]], dtype=np.float32)
    model = HybridClassifier(**model_kwargs)
    model.fit(data["adj_csr"], data["features"].toarray().astype(np.float32), labels, train_nodes)
    preds = model.predict(val_nodes)
    acc = accuracy_score(labels[val_nodes], preds)
    return {"name": "Hybrid", "accuracy": float(acc), "kind": "hybrid"}, model


def _train_ensemble_candidate(
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    raw_features: np.ndarray,
    enhanced_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, list[ScaledLGBM]]:
    _, weight_lookup = _balanced_sample_weights(labels[train_nodes])
    train_weights = np.asarray([weight_lookup[int(y)] for y in labels[train_nodes]], dtype=np.float32)

    specs = [
        ("raw", raw_features, {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.05}),
        ("enhanced", enhanced_features, {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03}),
        ("regularized", enhanced_features, {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.01}),
    ]

    members: list[ScaledLGBM] = []
    val_probs = None
    for _, features, kwargs in specs:
        candidate = ScaledLGBM(
            name="ensemble_member",
            model=LightGBMModel(**kwargs),
            scaler=StandardScaler(),
            feature_matrix=features,
        )
        candidate.fit(train_nodes, labels, sample_weight=train_weights)
        members.append(candidate)
        member_probs = candidate.predict_proba(val_nodes)
        val_probs = member_probs if val_probs is None else val_probs + member_probs

    val_probs /= len(members)
    preds = val_probs.argmax(axis=1)
    acc = accuracy_score(labels[val_nodes], preds)
    return {"name": "LGBM Ensemble", "accuracy": float(acc), "kind": "ensemble"}, members


def _refit_and_predict(best_candidate: dict, data: dict, train_nodes: np.ndarray, test_idx: np.ndarray, raw_features, enhanced_features) -> np.ndarray:
    labels = data["labels"]
    _, weight_lookup = _balanced_sample_weights(labels[train_nodes])
    train_weights = np.asarray([weight_lookup[int(y)] for y in labels[train_nodes]], dtype=np.float32)

    if best_candidate["kind"] == "hybrid":
        model = HybridClassifier()
        model.fit(data["adj_csr"], data["features"].toarray().astype(np.float32), labels, train_nodes)
        return model.predict(test_idx)

    if best_candidate["kind"] == "ensemble":
        refs = [
            (raw_features, {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.05}),
            (enhanced_features, {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03}),
            (enhanced_features, {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.01}),
        ]
        probs = None
        for features, kwargs in refs:
            candidate = ScaledLGBM(
                name="ensemble_member",
                model=LightGBMModel(**kwargs),
                scaler=StandardScaler(),
                feature_matrix=features,
            )
            candidate.fit(train_nodes, labels, sample_weight=train_weights)
            cur = candidate.predict_proba(test_idx)
            probs = cur if probs is None else probs + cur
        probs /= len(refs)
        return probs.argmax(axis=1)

    feature_matrix = raw_features if best_candidate["name"] == "Raw LightGBM" else enhanced_features
    candidate = ScaledLGBM(
        name=best_candidate["name"],
        model=LightGBMModel(n_estimators=500, max_depth=7, learning_rate=0.03),
        scaler=StandardScaler(),
        feature_matrix=feature_matrix,
    )
    candidate.fit(train_nodes, labels, sample_weight=train_weights)
    return candidate.predict(test_idx)


def main():
    print("=" * 60)
    print("AFAC2026 - Classification task (A1)")
    print("=" * 60)
    total_start = time.time()

    print("\n[1] Loading data...")
    data = load_classification(config.CLS_NPZ)
    print(f"  nodes: {data['num_nodes']}")
    print(f"  features: {data['num_features']}")
    print(f"  classes: {data['num_classes']}")
    print(f"  train nodes: {len(data['train_idx'])}")
    print(f"  test nodes: {len(data['test_idx'])}")

    print("\n[2] Analyzing data...")
    data_profile = analyze_data(data)
    print(f"  sparsity: {data_profile['sparsity']:.1%}")
    print(f"  avg degree: {data_profile['avg_degree']:.2f}")
    print(f"  isolated ratio: {data_profile['isolated_ratio']:.1%}")
    print(f"  class imbalance: {data_profile['imbalance']:.1f}x")

    qwen_client = None
    if HAS_QWEN:
        try:
            qwen_client = QwenClient()
            if qwen_client.available:
                print("\n[Qwen] connected")
        except Exception as exc:
            print(f"\n[Qwen] init failed: {exc}")

    raw_features, enhanced_features = _build_graph_features(data)
    train_idx = np.asarray(data["train_idx"], dtype=int)
    labels = data["labels"]

    # Phase 1: 使用固定验证集划分
    train_nodes, val_nodes = get_fixed_split(
        train_idx, labels, val_ratio=0.2, seed=42,
    )

    # Phase 2: 计算类别平衡权重（用于日志记录）
    _ = compute_class_weights(labels[train_nodes])

    results: list[dict] = []
    print("\n[3] Training candidates...")
    raw_result, raw_model = _train_lgbm_candidate(
        "Raw LightGBM",
        raw_features,
        train_nodes,
        val_nodes,
        labels,
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05},
    )
    results.append(raw_result)
    print(f"  Raw LightGBM: {raw_result['accuracy']:.4f}")

    enhanced_result, enhanced_model = _train_lgbm_candidate(
        "Enhanced LightGBM",
        enhanced_features,
        train_nodes,
        val_nodes,
        labels,
        {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03},
    )
    results.append(enhanced_result)
    print(f"  Enhanced LightGBM: {enhanced_result['accuracy']:.4f}")

    ensemble_result, ensemble_model = _train_ensemble_candidate(
        train_nodes,
        val_nodes,
        raw_features,
        enhanced_features,
        labels,
    )
    results.append(ensemble_result)
    print(f"  LGBM Ensemble: {ensemble_result['accuracy']:.4f}")

    hybrid_result, hybrid_model = _train_hybrid_candidate(
        train_nodes,
        val_nodes,
        data,
        labels,
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05},
    )
    results.append(hybrid_result)
    print(f"  Hybrid: {hybrid_result['accuracy']:.4f}")

    print("\n[4] Qwen analysis...")
    analysis = qwen_analyze(qwen_client, data_profile, results)
    print(f"  {analysis if analysis else 'skipped'}")

    best = max(results, key=lambda item: item["accuracy"])
    print("\n[5] Best candidate")
    print(f"  {best['name']}")
    print(f"  val accuracy: {best['accuracy']:.4f}")

    # Phase 10: 排行榜记录
    lb = Leaderboard()
    for r in results:
        lb.add(
            model_name=r["name"],
            feature_dim=enhanced_features.shape[1],
            val_acc=r["accuracy"],
        )
    lb.display()

    print("\n[6] Generating submission...")
    test_idx = np.asarray(data["test_idx"], dtype=int)
    predictions = _refit_and_predict(best, data, train_idx, test_idx, raw_features, enhanced_features)

    output_path = os.path.join(config.OUTPUT_DIR, "A1.csv")
    df = pd.DataFrame({"test_idx": test_idx, "label": predictions})
    df.to_csv(output_path, index=False)
    print(f"  saved: {output_path}")

    validate_A1(output_path, len(test_idx), data["num_classes"])

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  best val accuracy: {best['accuracy']:.4f}")
    print(f"  total time: {total_time:.1f}s")
    print(f"  submission: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

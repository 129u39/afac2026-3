"""自主科研 Agent：分类任务迭代优化。"""

import os
import sys
import json
import time
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config
from data_loader import load_classification
from submit import validate_A1
from analysis.research_agent import ResearchAgent, ExperimentResult


def _compute_aggregated_features(adj, feat_dense, K=2):
    """计算多跳图聚合特征（SIGN风格）。

    Args:
        adj: CSR 邻接矩阵
        feat_dense: 稠密特征矩阵 (N, D)
        K: 最大跳数

    返回:
        agg_features: (N, D * K) 所有跳数的聚合特征
    """
    from scipy.sparse import diags
    N = feat_dense.shape[0]
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg > 0] = np.power(deg[deg > 0], -0.5)
    D_inv_sqrt = diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    aggs = []
    x = feat_dense.copy()
    for k in range(K):
        x = adj_norm @ x
        if hasattr(x, 'toarray'):
            x = x.toarray()
        aggs.append(x)
    return np.concatenate(aggs, axis=1)


def _load_data():
    """加载并预处理数据（兼容 GNN / LGB / Hybrid）。"""
    import numpy as np
    from scipy.sparse import csr_matrix, diags

    d = np.load(config.CLS_NPZ)
    adj = csr_matrix(
        (d["adj_data"], d["adj_indices"], d["adj_indptr"]),
        shape=tuple(d["adj_shape"]),
    )
    features_sparse = csr_matrix(
        (d["attr_data"], d["attr_indices"], d["attr_indptr"]),
        shape=tuple(d["attr_shape"]),
    )
    feat_dense = features_sparse.toarray().astype(np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()

    # 预计算图聚合特征：A@X, A^2@X
    agg_feat = _compute_aggregated_features(adj, feat_dense, K=2)

    # 对数度特征
    log_deg = np.log1p(deg).reshape(-1, 1)

    # 增强特征：原始 + 1跳聚合 + 2跳聚合 + 度
    hop1 = agg_feat[:, :feat_dense.shape[1]]
    hop2 = agg_feat[:, feat_dense.shape[1]:]
    enhanced_features = np.concatenate([feat_dense, hop1, hop2, log_deg], axis=1)

    return {
        # Hybrid 用
        "adj": adj,
        "features_dense": feat_dense,
        "enhanced_features": enhanced_features,  # 增强特征（含图聚合）
        "degrees": deg,
        "log_degrees": log_deg,
        # GNN / LGB 向后兼容
        "adj_csr": adj,
        "features": features_sparse,
        # 通用
        "labels": d["labels"],
        "train_idx": d["train_idx"],
        "test_idx": d["test_idx"],
        "num_nodes": d["adj_shape"][0],
        "num_features": feat_dense.shape[1],
        "num_classes": int(d["labels"].max()) + 1,
    }

# LLM 客户端
try:
    from llm.client import QwenClient
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False


def run_classification_experiment(config: dict, data=None, device=None) -> ExperimentResult:
    """执行单次分类实验。"""
    if data is None:
        data = load_classification(config.CLS_NPZ)
    if device is None:
        from models.utils import get_device
        device = get_device()

    model_type = config.get("model_type", "GraphSAGE")

    # 根据模型类型选择不同的评估方式
    if model_type == "Ensemble":
        return _run_ensemble(config, data)
    elif model_type == "Hybrid":
        return _run_hybrid(config, data)
    elif model_type == "LightGBM":
        return _run_lgb(config, data)
    elif model_type in ("GraphSAGE", "GCN", "GAT"):
        return _run_gnn(config, data, device)
    else:
        return _run_ensemble(config, data)


def _run_hybrid(config: dict, data) -> ExperimentResult:
    """混合分类器：图聚合特征 + LightGBM + 软融合邻居传播。

    策略：
    1. LightGBM 用增强特征（原始+1跳聚合+2跳聚合+度）训练
    2. 邻居传播软融合：用置信度平滑加权 LGBM 和邻居标签分布
    3. 2跳传播：当1跳邻居不足时使用2跳
    """
    from lightgbm import LGBMClassifier
    from collections import Counter
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    adj = data["adj"]
    enhanced = data["enhanced_features"]
    labels = data["labels"]
    train_idx = np.array(data["train_idx"]).astype(int)

    # 单次切分
    train_nodes, val_nodes = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    # LightGBM 训练（使用增强特征）
    X_train = enhanced[train_nodes]
    y_train = labels[train_nodes]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    model = LGBMClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.03),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        min_child_samples=config.get("min_child_samples", 20),
        verbose=-1,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    # 混合预测（软融合）
    X_val_s = scaler.transform(enhanced[val_nodes])
    y_val = labels[val_nodes]

    lgb_proba = model.predict_proba(X_val_s)

    # 预计算所有训练节点的1跳邻居集合（加速邻居查询）
    train_set = set(train_nodes)
    neighbor_cache = {}
    def _get_neighbors(node):
        if node not in neighbor_cache:
            nbrs = adj[node].nonzero()[1]
            valid = [n for n in nbrs if n in train_set]
            neighbor_cache[node] = valid
        return neighbor_cache[node]

    hybrid_preds = []
    for i, node in enumerate(val_nodes):
        valid_neighbors = _get_neighbors(node)

        # 2跳邻居（仅当1跳不足时）
        if len(valid_neighbors) < 3:
            seen = set(valid_neighbors)
            for n in _get_neighbors(node):
                for n2 in _get_neighbors(n):
                    if n2 not in seen:
                        seen.add(n2)
            valid_neighbors = list(seen)

        if len(valid_neighbors) > 0:
            neighbor_labels = labels[valid_neighbors]
            counts = Counter(neighbor_labels)

            neighbor_dist = np.zeros(lgb_proba.shape[1])
            for cls, cnt in counts.items():
                neighbor_dist[cls] = cnt / len(valid_neighbors)

            majority_cls, majority_cnt = counts.most_common(1)[0]
            neighbor_confidence = majority_cnt / len(valid_neighbors)

            blend_weight = neighbor_confidence ** 1.5
            blended_proba = blend_weight * neighbor_dist + (1 - blend_weight) * lgb_proba[i]
            hybrid_preds.append(blended_proba.argmax())
        else:
            hybrid_preds.append(lgb_proba[i].argmax())

    acc = accuracy_score(y_val, hybrid_preds)

    return ExperimentResult(
        config=config,
        metric=acc,
        metric_name="val_accuracy",
        train_time=0.0,
    )


def _run_lgb(config: dict, data) -> ExperimentResult:
    """LightGBM 分类（使用增强特征）。"""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    enhanced = data["enhanced_features"]
    labels = data["labels"]
    train_idx = np.array(data["train_idx"]).astype(int)

    X = enhanced[train_idx]
    y = labels[train_idx]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    model = LGBMClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.03),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        min_child_samples=config.get("min_child_samples", 20),
        verbose=-1,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)
    preds = model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)

    return ExperimentResult(
        config=config,
        metric=acc,
        metric_name="val_accuracy",
        train_time=0.0,
    )


def _run_gnn(config: dict, data, device) -> ExperimentResult:
    """GNN 分类。"""
    from models.gnn_classifier import GNNClassifier, train_gnn
    from data_loader import classification_to_pyg
    from evaluate import evaluate_classification

    pyg_data = classification_to_pyg(data, device)
    model_type = config.get("model_type", "GraphSAGE")

    model = GNNClassifier(
        in_dim=data["num_features"],
        hidden_dim=config.get("hidden_dim", 128),
        num_classes=data["num_classes"],
        num_layers=config.get("num_layers", 3),
        model_type=model_type,
        dropout=config.get("dropout", 0.2),
    ).to(device)

    train_gnn(
        model, pyg_data,
        lr=config.get("lr", 0.005),
        weight_decay=config.get("weight_decay", 5e-4),
        epochs=config.get("epochs", 100),
        patience=config.get("patience", 15),
    )

    eval_result = evaluate_classification(model, pyg_data, data["train_idx"], data["labels"])

    return ExperimentResult(
        config=config,
        metric=eval_result["val_accuracy"],
        metric_name="val_accuracy",
        train_time=0.0,
    )


def _run_ensemble(config: dict, data) -> ExperimentResult:
    """多模型 LightGBM 集成：不同特征子集的 LightGBM 模型集成。

    避免 GNN 的图信息泄露问题，使用纯特征工程方法。
    模型：
    1. LightGBM (enhanced: raw + 1hop + 2hop + degree)
    2. LightGBM (raw features only)
    3. LightGBM (graph features: 1hop + 2hop + degree)
    4. LightGBM (enhanced, different hyperparams)
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    enhanced = data["enhanced_features"]
    features_dense = data["features_dense"]
    labels = data["labels"]
    train_idx = np.array(data["train_idx"]).astype(int)
    num_classes = data["num_classes"]

    # 构造图聚合特征 (1hop + 2hop + degree)
    feat_dim = features_dense.shape[1]
    graph_features = enhanced[:, feat_dim:]  # 1hop + 2hop + degree

    # 划分 train/val
    train_nodes, val_nodes = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )
    y_train = labels[train_nodes]
    y_val = labels[val_nodes]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    probas = []
    accs = []

    # === 模型1: LightGBM (enhanced features) ===
    scaler1 = StandardScaler()
    X1_train = scaler1.fit_transform(enhanced[train_nodes])
    X1_val = scaler1.transform(enhanced[val_nodes])
    m1 = LGBMClassifier(
        n_estimators=config.get("n_estimators", 500),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.03),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        min_child_samples=config.get("min_child_samples", 20),
        verbose=-1,
    )
    m1.fit(X1_train, y_train, sample_weight=sample_weights)
    p1 = m1.predict_proba(X1_val)
    probas.append(p1)
    accs.append(accuracy_score(y_val, p1.argmax(axis=1)))

    # === 模型2: LightGBM (raw features only) ===
    scaler2 = StandardScaler()
    X2_train = scaler2.fit_transform(features_dense[train_nodes])
    X2_val = scaler2.transform(features_dense[val_nodes])
    m2 = LGBMClassifier(
        n_estimators=config.get("n_estimators", 500),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.03),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        min_child_samples=config.get("min_child_samples", 20),
        verbose=-1,
    )
    m2.fit(X2_train, y_train, sample_weight=sample_weights)
    p2 = m2.predict_proba(X2_val)
    probas.append(p2)
    accs.append(accuracy_score(y_val, p2.argmax(axis=1)))

    # === 模型3: LightGBM (graph features) ===
    scaler3 = StandardScaler()
    X3_train = scaler3.fit_transform(graph_features[train_nodes])
    X3_val = scaler3.transform(graph_features[val_nodes])
    m3 = LGBMClassifier(
        n_estimators=config.get("n_estimators", 500),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.03),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 0.1),
        min_child_samples=config.get("min_child_samples", 20),
        verbose=-1,
    )
    m3.fit(X3_train, y_train, sample_weight=sample_weights)
    p3 = m3.predict_proba(X3_val)
    probas.append(p3)
    accs.append(accuracy_score(y_val, p3.argmax(axis=1)))

    # === 模型4: LightGBM (enhanced, stronger regularization) ===
    scaler4 = StandardScaler()
    X4_train = scaler4.fit_transform(enhanced[train_nodes])
    X4_val = scaler4.transform(enhanced[val_nodes])
    m4 = LGBMClassifier(
        n_estimators=config.get("n_estimators", 800),
        max_depth=config.get("max_depth", 5),
        learning_rate=config.get("learning_rate", 0.01),
        subsample=config.get("subsample", 0.7),
        colsample_bytree=config.get("colsample_bytree", 0.7),
        reg_alpha=config.get("reg_alpha", 1.0),
        reg_lambda=config.get("reg_lambda", 1.0),
        min_child_samples=config.get("min_child_samples", 30),
        verbose=-1,
    )
    m4.fit(X4_train, y_train, sample_weight=sample_weights)
    p4 = m4.predict_proba(X4_val)
    probas.append(p4)
    accs.append(accuracy_score(y_val, p4.argmax(axis=1)))

    # === 加权集成 ===
    accs = np.array(accs)
    weights = accs / accs.sum()

    ensemble_proba = sum(w * p for w, p in zip(weights, probas))
    ensemble_acc = accuracy_score(y_val, ensemble_proba.argmax(axis=1))

    print(f"  模型准确率: LGB-enh={accs[0]:.4f}, LGB-raw={accs[1]:.4f}, "
          f"LGB-graph={accs[2]:.4f}, LGB-reg={accs[3]:.4f}")
    print(f"  集成准确率: {ensemble_acc:.4f}")
    print(f"  权重: {weights}")

    return ExperimentResult(
        config=config,
        metric=ensemble_acc,
        metric_name="val_accuracy",
        train_time=0.0,
    )


def main():
    print("=" * 60)
    print("AFAC2026 - 自主科研 Agent（分类任务）")
    print("=" * 60)

    # 初始化 LLM
    llm_client = None
    if HAS_QWEN:
        try:
            llm_client = QwenClient()
            if llm_client.available:
                print(f"[Qwen] 已连接")
        except Exception:
            pass

    # 加载数据
    data = _load_data()

    # 创建 Agent
    agent = ResearchAgent(
        task_type="classification",
        run_fn=lambda config: run_classification_experiment(config, data=data),
        llm_client=llm_client,
        memory_path="output/research_memory_classification.json",
    )

    # 初始配置（多模型集成）
    initial_config = {
        "model_type": "Ensemble",
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "hidden_dim": 256,
        "gcnii_layers": 16,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200,
        "patience": 30,
    }

    # 运行
    result = agent.run(initial_config)

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"  最佳 val_accuracy: {result.get('best_metric', 0.0):.4f}")
    print(f"  最佳配置: {result.get('best_config', {})}")
    print(f"  总轮次: {result.get('num_rounds', 0)}")

    # 生成提交
    best_config = result["best_config"]
    if best_config:
        print("\n生成最终提交...")
        import pandas as pd

        model_type = best_config.get("model_type", "Ensemble")

        if model_type == "Ensemble":
            # 集成提交：多个 LightGBM 模型（不同特征子集）
            from lightgbm import LGBMClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.utils.class_weight import compute_class_weight

            enhanced = data["enhanced_features"]
            features_dense = data["features_dense"]
            labels = data["labels"]
            train_idx = np.array(data["train_idx"]).astype(int)
            test_idx = np.array(data["test_idx"]).astype(int)

            feat_dim = features_dense.shape[1]
            graph_features = enhanced[:, feat_dim:]

            y_train = labels[train_idx]
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = np.array([class_weights[y] for y in y_train])

            probas = []

            # Model 1: LightGBM (enhanced)
            s1 = StandardScaler()
            X1_train = s1.fit_transform(enhanced[train_idx])
            X1_test = s1.transform(enhanced[test_idx])
            m1 = LGBMClassifier(
                n_estimators=best_config.get("n_estimators", 500),
                max_depth=best_config.get("max_depth", 7),
                learning_rate=best_config.get("learning_rate", 0.03),
                subsample=best_config.get("subsample", 0.8),
                colsample_bytree=best_config.get("colsample_bytree", 0.8),
                reg_alpha=best_config.get("reg_alpha", 0.1),
                reg_lambda=best_config.get("reg_lambda", 0.1),
                min_child_samples=best_config.get("min_child_samples", 20),
                verbose=-1,
            )
            m1.fit(X1_train, y_train, sample_weight=sample_weights)
            probas.append(m1.predict_proba(X1_test))

            # Model 2: LightGBM (raw)
            s2 = StandardScaler()
            X2_train = s2.fit_transform(features_dense[train_idx])
            X2_test = s2.transform(features_dense[test_idx])
            m2 = LGBMClassifier(
                n_estimators=best_config.get("n_estimators", 500),
                max_depth=best_config.get("max_depth", 7),
                learning_rate=best_config.get("learning_rate", 0.03),
                subsample=best_config.get("subsample", 0.8),
                colsample_bytree=best_config.get("colsample_bytree", 0.8),
                reg_alpha=best_config.get("reg_alpha", 0.1),
                reg_lambda=best_config.get("reg_lambda", 0.1),
                min_child_samples=best_config.get("min_child_samples", 20),
                verbose=-1,
            )
            m2.fit(X2_train, y_train, sample_weight=sample_weights)
            probas.append(m2.predict_proba(X2_test))

            # Model 3: LightGBM (graph features)
            s3 = StandardScaler()
            X3_train = s3.fit_transform(graph_features[train_idx])
            X3_test = s3.transform(graph_features[test_idx])
            m3 = LGBMClassifier(
                n_estimators=best_config.get("n_estimators", 500),
                max_depth=best_config.get("max_depth", 7),
                learning_rate=best_config.get("learning_rate", 0.03),
                subsample=best_config.get("subsample", 0.8),
                colsample_bytree=best_config.get("colsample_bytree", 0.8),
                reg_alpha=best_config.get("reg_alpha", 0.1),
                reg_lambda=best_config.get("reg_lambda", 0.1),
                min_child_samples=best_config.get("min_child_samples", 20),
                verbose=-1,
            )
            m3.fit(X3_train, y_train, sample_weight=sample_weights)
            probas.append(m3.predict_proba(X3_test))

            # Model 4: LightGBM (enhanced, stronger regularization)
            s4 = StandardScaler()
            X4_train = s4.fit_transform(enhanced[train_idx])
            X4_test = s4.transform(enhanced[test_idx])
            m4 = LGBMClassifier(
                n_estimators=800,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                min_child_samples=30,
                verbose=-1,
            )
            m4.fit(X4_train, y_train, sample_weight=sample_weights)
            probas.append(m4.predict_proba(X4_test))

            # 等权重集成
            ensemble_proba = sum(probas) / len(probas)
            predictions = ensemble_proba.argmax(axis=1)

        elif model_type in ("Hybrid", "LightGBM"):
            from lightgbm import LGBMClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.utils.class_weight import compute_class_weight
            from collections import Counter

            adj = data["adj"]
            enhanced = data["enhanced_features"]
            labels = data["labels"]
            train_idx = np.array(data["train_idx"]).astype(int)
            test_idx = np.array(data["test_idx"]).astype(int)

            X_train = enhanced[train_idx]
            y_train = labels[train_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = np.array([class_weights[y] for y in y_train])

            model = LGBMClassifier(
                n_estimators=best_config.get("n_estimators", 300),
                max_depth=best_config.get("max_depth", 7),
                learning_rate=best_config.get("learning_rate", 0.03),
                subsample=best_config.get("subsample", 0.8),
                colsample_bytree=best_config.get("colsample_bytree", 0.8),
                reg_alpha=best_config.get("reg_alpha", 0.1),
                reg_lambda=best_config.get("reg_lambda", 0.1),
                min_child_samples=best_config.get("min_child_samples", 20),
                verbose=-1,
            )
            model.fit(X_train_s, y_train, sample_weight=sample_weights)

            if model_type == "Hybrid":
                X_test_s = scaler.transform(enhanced[test_idx])
                lgb_proba = model.predict_proba(X_test_s)
                test_preds = []
                for i, node in enumerate(test_idx):
                    neighbors = adj[node].nonzero()[1]
                    valid_neighbors = np.intersect1d(neighbors, train_idx)

                    if len(valid_neighbors) < 3:
                        second_hop = set()
                        for n in neighbors:
                            n2 = adj[n].nonzero()[1]
                            second_hop.update(np.intersect1d(n2, train_idx))
                        valid_neighbors = np.union1d(valid_neighbors, list(second_hop))

                    if len(valid_neighbors) > 0:
                        valid_neighbors = valid_neighbors.astype(int)
                        neighbor_labels = labels[valid_neighbors]
                        counts = Counter(neighbor_labels)

                        neighbor_dist = np.zeros(lgb_proba.shape[1])
                        for cls, cnt in counts.items():
                            neighbor_dist[cls] = cnt / len(valid_neighbors)

                        _, majority_cnt = counts.most_common(1)[0]
                        neighbor_confidence = majority_cnt / len(valid_neighbors)
                        blend_weight = neighbor_confidence ** 1.5
                        blended_proba = blend_weight * neighbor_dist + (1 - blend_weight) * lgb_proba[i]
                        test_preds.append(blended_proba.argmax())
                    else:
                        test_preds.append(lgb_proba[i].argmax())
                predictions = np.array(test_preds)
            else:
                X_test_s = scaler.transform(enhanced[test_idx])
                predictions = model.predict(X_test_s)

        else:
            # GNN 提交
            from models.utils import get_device
            device = get_device()
            from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn
            from data_loader import classification_to_pyg

            pyg_data = classification_to_pyg(data, device)
            model = GNNClassifier(
                in_dim=data["num_features"],
                hidden_dim=best_config.get("hidden_dim", 128),
                num_classes=data["num_classes"],
                num_layers=best_config.get("num_layers", 3),
                model_type=model_type,
                dropout=best_config.get("dropout", 0.2),
            ).to(device)
            train_gnn(model, pyg_data,
                      lr=best_config.get("lr", 0.005),
                      weight_decay=best_config.get("weight_decay", 5e-4),
                      epochs=best_config.get("epochs", 100),
                      patience=best_config.get("patience", 15))
            predictions = predict_gnn(model, pyg_data).numpy()

        df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions})
        output_path = os.path.join(config.OUTPUT_DIR, "A1.csv")
        df.to_csv(output_path, index=False)
        print(f"  已保存: {output_path} ({len(df)} 行)")
        validate_A1(output_path, len(data["test_idx"]), data["num_classes"])

    print("=" * 60)


if __name__ == "__main__":
    main()

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

# LLM 客户端
try:
    from llm.client import QwenClient
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False


def run_classification_experiment(config: dict, data=None, device=None) -> ExperimentResult:
    """执行单次分类实验。"""
    if data is None:
        data = load_classification("data/A分类/A分类/A1.npz")
    if device is None:
        from models.utils import get_device
        device = get_device()

    model_type = config.get("model_type", "GraphSAGE")

    # 根据模型类型选择不同的评估方式
    if model_type == "Hybrid":
        return _run_hybrid(config, data)
    elif model_type == "LightGBM":
        return _run_lgb(config, data)
    elif model_type in ("GraphSAGE", "GCN", "GAT"):
        return _run_gnn(config, data, device)
    else:
        return _run_hybrid(config, data)


def _run_hybrid(config: dict, data) -> ExperimentResult:
    """混合分类器：邻居标签传播 + LightGBM。"""
    from lightgbm import LGBMClassifier
    from scipy.sparse import csr_matrix, diags
    from collections import Counter

    features = data["features"].toarray()
    labels = data["labels"]
    train_idx = data["train_idx"]
    adj = csr_matrix((data["adj_data"], data["adj_indices"], data["adj_indptr"]),
                     shape=tuple(data["adj_shape"]))

    # 训练/验证划分
    X_train, X_val, y_train, y_val = train_test_split(
        features[train_idx], labels[train_idx],
        test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # LightGBM 训练
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    model = LGBMClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 8),
        learning_rate=config.get("learning_rate", 0.05),
        verbose=-1,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    # 混合预测
    hybrid_preds = []
    for i, (row_idx, true_label) in enumerate(zip(
        train_idx[X_train.shape[0]:], y_val
    )):
        neighbors = adj[row_idx].nonzero()[1]
        if len(neighbors) > 0:
            neighbor_labels = labels[neighbors]
            counts = Counter(neighbor_labels)
            majority, majority_count = counts.most_common(1)[0]
            if majority_count / len(neighbors) > 0.7:
                hybrid_preds.append(majority)
            else:
                hybrid_preds.append(model.predict(X_val_s[i:i+1])[0])
        else:
            hybrid_preds.append(model.predict(X_val_s[i:i+1])[0])

    acc = accuracy_score(y_val, hybrid_preds)

    return ExperimentResult(
        config=config,
        metric=acc,
        metric_name="val_accuracy",
        train_time=0.0,
    )


def _run_lgb(config: dict, data) -> ExperimentResult:
    """LightGBM 分类。"""
    from lightgbm import LGBMClassifier

    features = data["features"].toarray()
    labels = data["labels"]
    train_idx = data["train_idx"]

    X = features[train_idx]
    y = labels[train_idx]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LGBMClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        verbose=-1,
    )
    model.fit(X_train_s, y_train)
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

    # 创建 Agent
    agent = ResearchAgent(
        task_type="classification",
        run_fn=run_classification_experiment,
        llm_client=llm_client,
        memory_path="output/research_memory_classification.json",
    )

    # 初始配置
    initial_config = {
        "model_type": "GraphSAGE",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "epochs": 100,
        "patience": 15,
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
        final_result = run_classification_experiment(best_config)

        # 使用最佳模型生成提交
        data = load_classification("data/A分类/A分类/A1.npz")
        from models.utils import get_device
        device = get_device()
        from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn
        from data_loader import classification_to_pyg

        pyg_data = classification_to_pyg(data, device)
        model_type = best_config.get("model_type", "GraphSAGE")

        model = GNNClassifier(
            in_dim=data["num_features"],
            hidden_dim=best_config.get("hidden_dim", 128),
            num_classes=data["num_classes"],
            num_layers=best_config.get("num_layers", 3),
            model_type=model_type,
            dropout=best_config.get("dropout", 0.2),
        ).to(device)

        train_gnn(model, pyg_data, lr=best_config.get("lr", 0.005),
                  weight_decay=best_config.get("weight_decay", 5e-4),
                  epochs=best_config.get("epochs", 100),
                  patience=best_config.get("patience", 15))

        predictions = predict_gnn(model, pyg_data)

        import pandas as pd
        df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions.numpy()})
        output_path = os.path.join(config.OUTPUT_DIR, "A1.csv")
        df.to_csv(output_path, index=False)
        print(f"  已保存: {output_path} ({len(df)} 行)")
        validate_A1(output_path, len(data["test_idx"]), data["num_classes"])

    print("=" * 60)


if __name__ == "__main__":
    main()

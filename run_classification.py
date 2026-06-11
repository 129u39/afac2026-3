"""运行产品分类任务 — Qwen 指导自动化实验。"""

import os
import time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config
from data_loader import load_classification
from submit import validate_A1

# Qwen 客户端
try:
    from llm.client import QwenClient
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

# 模型
from models.hybrid_classifier import HybridClassifier
from models.lightgbm_model import LightGBMModel


def analyze_data(data):
    """分析数据特征。"""
    features = data["features"].toarray()
    labels = data["labels"]
    train_idx = data["train_idx"]
    adj = data["adj_csr"]

    # 基本统计
    deg = np.array(adj.sum(axis=1)).flatten()
    isolated_ratio = (deg == 0).sum() / len(deg)

    # 类别分布
    train_labels = labels[train_idx]
    unique, counts = np.unique(train_labels, return_counts=True)
    imbalance = counts.max() / counts.min()

    return {
        "num_nodes": data["num_nodes"],
        "num_features": data["num_features"],
        "num_classes": data["num_classes"],
        "sparsity": (features == 0).sum() / features.size,
        "avg_degree": deg.mean(),
        "isolated_ratio": isolated_ratio,
        "imbalance": imbalance,
        "class_distribution": {int(u): int(c) for u, c in zip(unique, counts)},
    }


def qwen_analyze(qwen_client, data_profile, current_results):
    """请求 Qwen 分析实验结果并给出建议。"""
    if not qwen_client or not qwen_client.available:
        return None

    prompt = f"""## 数据画像
- 节点数: {data_profile['num_nodes']}
- 特征维度: {data_profile['num_features']}
- 类别数: {data_profile['num_classes']}
- 特征稀疏度: {data_profile['sparsity']:.1%}
- 平均度: {data_profile['avg_degree']:.2f}
- 孤立节点比例: {data_profile['isolated_ratio']:.1%}
- 类别不平衡: {data_profile['imbalance']:.1f}x

## 当前实验结果
{chr(10).join(f"- {r['name']}: {r['accuracy']:.4f}" for r in current_results)}

## 任务
请分析当前结果，给出下一步实验建议。考虑：
1. 哪种策略最有效？
2. 还有哪些改进空间？
3. 建议的超参数调整？

请简洁回答（不超过200字）。"""

    try:
        response = qwen_client.chat(
            "你是一个机器学习专家，专注于图神经网络和特征工程。",
            prompt,
        )
        return response
    except Exception as e:
        return f"Qwen 分析失败: {e}"


def main():
    print("=" * 60)
    print("AFAC2026 - 产品分类任务 (A1)")
    print("=" * 60)

    total_start = time.time()

    # 加载数据
    print("\n[1] 加载数据...")
    data = load_classification(config.CLS_NPZ)
    print(f"  节点数: {data['num_nodes']}")
    print(f"  特征维度: {data['num_features']}")
    print(f"  类别数: {data['num_classes']}")
    print(f"  训练节点: {len(data['train_idx'])}")
    print(f"  测试节点: {len(data['test_idx'])}")

    # 数据分析
    print("\n[2] 数据分析...")
    data_profile = analyze_data(data)
    print(f"  特征稀疏度: {data_profile['sparsity']:.1%}")
    print(f"  平均度: {data_profile['avg_degree']:.2f}")
    print(f"  孤立节点: {data_profile['isolated_ratio']:.1%}")
    print(f"  类别不平衡: {data_profile['imbalance']:.1f}x")

    # 初始化 Qwen
    qwen_client = None
    if HAS_QWEN:
        try:
            qwen_client = QwenClient()
            if qwen_client.available:
                print(f"\n[Qwen] 已连接 (qwen-turbo)")
            else:
                print(f"\n[Qwen] API Key 未设置，使用规则模式")
        except Exception as e:
            print(f"\n[Qwen] 初始化失败: {e}")

    # 实验结果记录
    results = []

    # ── 实验 1: LightGBM 基线 ──
    print("\n" + "=" * 60)
    print("[实验 1] LightGBM 基线")
    print("=" * 60)

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

    model = LightGBMModel(n_estimators=500, max_depth=8)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    results.append({"name": "LightGBM baseline", "accuracy": acc})
    print(f"  准确率: {acc:.4f}")

    # ── 实验 2: LightGBM + 类别加权 ──
    print("\n" + "=" * 60)
    print("[实验 2] LightGBM + 类别加权")
    print("=" * 60)

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    model = LightGBMModel(n_estimators=500, max_depth=8)
    model.fit(X_train_s, y_train)
    from lightgbm import LGBMClassifier
    model.model = LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, verbose=-1)
    model.model.fit(X_train_s, y_train, sample_weight=sample_weights)
    preds = model.model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    results.append({"name": "LightGBM + class_weight", "accuracy": acc})
    print(f"  准确率: {acc:.4f}")

    # ── 实验 3: 混合分类器 ──
    print("\n" + "=" * 60)
    print("[实验 3] 混合分类器 (LightGBM + 邻居传播)")
    print("=" * 60)

    adj = data["adj_csr"]
    hybrid = HybridClassifier(n_estimators=500, max_depth=8)
    hybrid.fit(adj, features, labels, train_idx)

    # 验证
    train_idx_sub, val_idx_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.2, random_state=42, stratify=y
    )
    val_nodes = train_idx[val_idx_sub]
    val_labels = labels[val_nodes]

    hybrid_preds = hybrid.predict(val_nodes)
    acc = accuracy_score(val_labels, hybrid_preds)
    results.append({"name": "Hybrid (LightGBM + Neighbor)", "accuracy": acc})
    print(f"  准确率: {acc:.4f}")

    # ── Qwen 分析 ──
    print("\n" + "=" * 60)
    print("[Qwen] 分析实验结果")
    print("=" * 60)

    analysis = qwen_analyze(qwen_client, data_profile, results)
    if analysis:
        print(f"  {analysis}")
    else:
        print("  跳过（Qwen 不可用）")

    # ── 选择最佳方案 ──
    print("\n" + "=" * 60)
    print("[结果] 选择最佳方案")
    print("=" * 60)

    best = max(results, key=lambda r: r["accuracy"])
    print(f"  最佳方案: {best['name']}")
    print(f"  最佳准确率: {best['accuracy']:.4f}")

    # ── 生成提交文件 ──
    print("\n" + "=" * 60)
    print("[提交] 生成 A1.csv")
    print("=" * 60)

    test_idx = data["test_idx"]
    final_preds = hybrid.predict(test_idx)

    import pandas as pd
    output_path = os.path.join(config.OUTPUT_DIR, "A1.csv")
    df = pd.DataFrame({"test_idx": test_idx, "label": final_preds})
    df.to_csv(output_path, index=False)
    print(f"  已保存: {output_path} ({len(df)} 行)")

    # 校验
    validate_A1(output_path, len(test_idx), data["num_classes"])

    # ── 总结 ──
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("最终总结")
    print("=" * 60)
    print(f"  分类准确率: {best['accuracy']:.4f}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

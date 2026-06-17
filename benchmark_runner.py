"""基准测试运行器：统一运行所有模型并输出排行榜。

Phase 1: Benchmark Framework
统一接口: fit() / predict() / evaluate()
"""
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import config
from data_loader import load_classification, classification_to_pyg
from utils.fixed_split import get_fixed_split
from losses.class_balanced import compute_class_weights
from leaderboard import Leaderboard


def evaluate_model(model, data, val_mask, labels, verbose=True):
    """统一评估接口。
    
    Args:
        model: 模型对象
        data: PyG Data 或特征矩阵
        val_mask: 验证掩码 (PyG) 或 val_idx (numpy)
        labels: 标签数组
        verbose: 是否打印

    Returns:
        {"accuracy": float, "macro_f1": float, "balanced_acc": float}
    """
    if isinstance(val_mask, torch.Tensor):
        # PyG 模型
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1).cpu().numpy()
            true = data.y[val_mask].cpu().numpy()
            pred = pred[val_mask.cpu().numpy()]
    else:
        # sklearn 模型
        if hasattr(model, "predict_proba"):
            pred = model.predict(data[val_mask])
        else:
            pred = model.predict(data[val_mask])
        true = labels[val_mask]

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(true, pred)
    if verbose:
        print(f"  acc={acc:.4f} macro_f1={f1:.4f} balanced_acc={bal:.4f}")
    return {"accuracy": acc, "macro_f1": f1, "balanced_acc": bal}


def run_benchmark(max_epochs=200, verbose=True):
    """运行所有模型基准测试。"""
    print("=" * 48)
    print("Benchmark Results")
    print("=" * 48)

    # 加载数据
    data = load_classification(config.CLS_NPZ)
    train_idx = np.asarray(data["train_idx"], dtype=int)
    labels = data["labels"]
    train_nodes, val_nodes = get_fixed_split(train_idx, labels)
    all_train = np.concatenate([train_nodes, val_nodes])  # 全量训练

    # PyG 数据
    device = torch.device("cpu")
    pyg_data = classification_to_pyg(data, device)

    # 验证掩码
    val_mask = torch.zeros(data["num_nodes"], dtype=torch.bool)
    val_mask[val_nodes] = True
    train_mask = torch.zeros(data["num_nodes"], dtype=torch.bool)
    val_mask[train_nodes] = True

    # 类别权重
    weights = compute_class_weights(labels[train_nodes])

    results = []
    lb = Leaderboard()

    # 1. MLP Baseline
    if verbose:
        print(f"\n[1] MLP Baseline")
    from models.mlp_baseline import MLPBaseline, train_mlp
    mlp = MLPBaseline(in_dim=data["num_features"], hidden_dim=512, num_classes=data["num_classes"])
    train_mlp(mlp, pyg_data, epochs=max_epochs, val_mask=val_mask, verbose=False)
    mlp_res = evaluate_model(mlp, pyg_data, val_mask, labels)
    results.append(("MLP", mlp_res))
    lb.add("MLP", feature_dim=data["num_features"], val_acc=mlp_res["accuracy"], macro_f1=mlp_res["macro_f1"])
    
    # 2. LightGBM (raw features)
    if verbose:
        print(f"\n[2] LightGBM")
    from models.lightgbm_model import LightGBMModel
    from sklearn.preprocessing import StandardScaler
    features = data["features"].toarray().astype(np.float32)
    lgb = LightGBMModel(n_estimators=500, max_depth=8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_nodes])
    lgb.fit(X_train, labels[train_nodes])
    X_val = scaler.transform(features[val_nodes])
    lgb_pred = lgb.model.predict(X_val)
    lgb_acc = accuracy_score(labels[val_nodes], lgb_pred)
    lgb_f1 = f1_score(labels[val_nodes], lgb_pred, average="macro", zero_division=0)
    lgb_bal = balanced_accuracy_score(labels[val_nodes], lgb_pred)
    if verbose:
        print(f"  acc={lgb_acc:.4f} macro_f1={lgb_f1:.4f} balanced_acc={lgb_bal:.4f}")
    results.append(("LightGBM", {"accuracy": lgb_acc, "macro_f1": lgb_f1, "balanced_acc": lgb_bal}))
    lb.add("LightGBM", feature_dim=data["num_features"], val_acc=lgb_acc, macro_f1=lgb_f1)

    # 3. GraphSAGE
    if verbose:
        print(f"\n[3] GraphSAGE")
    from models.gnn_classifier import GNNClassifier, train_gnn
    sage = GNNClassifier(in_dim=data["num_features"], hidden_dim=256, num_classes=data["num_classes"],
                         num_layers=3, model_type="GraphSAGE", dropout=0.5)
    train_gnn(sage, pyg_data, lr=5e-3, epochs=max_epochs, val_mask=val_mask, verbose=False, class_weights=weights)
    sage_res = evaluate_model(sage, pyg_data, val_mask, labels)
    results.append(("GraphSAGE", sage_res))
    lb.add("GraphSAGE", feature_dim=data["num_features"], val_acc=sage_res["accuracy"], macro_f1=sage_res["macro_f1"])

    # 4. GCNII
    if verbose:
        print(f"\n[4] GCNII")
    from models.gcnii import GCNII, train_gcnii
    gcnii = GCNII(in_dim=data["num_features"], hidden_dim=256, num_classes=data["num_classes"],
                  num_layers=16, alpha=0.2, theta=1.0)
    train_gcnii(gcnii, pyg_data, lr=0.01, epochs=max_epochs, val_mask=val_mask, verbose=False, class_weights=weights)
    gcnii_res = evaluate_model(gcnii, pyg_data, val_mask, labels)
    results.append(("GCNII", gcnii_res))
    lb.add("GCNII", feature_dim=data["num_features"], val_acc=gcnii_res["accuracy"], macro_f1=gcnii_res["macro_f1"])

    # 5. Label Propagation
    if verbose:
        print(f"\n[5] Label Propagation")
    from models.label_propagation import LabelPropagation
    lp = LabelPropagation(alpha=0.5)
    lp.fit(data["adj_csr"], labels, train_idx, val_nodes)
    lp_pred = lp.predict()
    lp_acc = accuracy_score(labels[val_nodes], lp_pred)
    lp_f1 = f1_score(labels[val_nodes], lp_pred, average="macro", zero_division=0)
    lp_bal = balanced_accuracy_score(labels[val_nodes], lp_pred)
    if verbose:
        print(f"  acc={lp_acc:.4f} macro_f1={lp_f1:.4f} balanced_acc={lp_bal:.4f}")
    results.append(("LabelPropagation", {"accuracy": lp_acc, "macro_f1": lp_f1, "balanced_acc": lp_bal}))
    lb.add("LabelPropagation", feature_dim=0, val_acc=lp_acc, macro_f1=lp_f1)

    # 输出排行榜
    print("\n" + "=" * 48)
    print("Final Results")
    print("=" * 48)
    for name, res in results:
        print(f"{name:20s}  acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}")
    print("=" * 48)

    print(f"\nPhase 6 LP Decision: LP acc={lp_acc:.4f}", end=" ")
    if lp_acc > 0.75:
        print("-> 图结构极强, 推荐 LP + GCNII")
    elif lp_acc < 0.60:
        print("-> 特征主导, 推荐 MLP + LightGBM")
    else:
        print("-> 混合路线")

    lb.display()
    return results


if __name__ == "__main__":
    run_benchmark()

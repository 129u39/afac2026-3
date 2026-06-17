"""数据诊断：输出数据集基本统计信息。"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def diagnose(data: dict, logger=None):
    """运行数据诊断并输出统计信息。

    Args:
        data: load_classification() 返回的数据字典
        logger: 可选的日志函数，默认 print
    """
    out = print if logger is None else logger

    adj = data["adj_csr"]
    feat = data["features"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    n_nodes = adj.shape[0]
    n_edges = adj.nnz
    avg_degree = n_edges / n_nodes if n_nodes > 0 else 0.0
    density = n_edges / (n_nodes * n_nodes) if n_nodes > 0 else 0.0

    out("=" * 60)
    out("[DATA] 数据集诊断")
    out("=" * 60)
    out(f"  nodes={n_nodes}")
    out(f"  edges={n_edges}")
    out(f"  avg_degree={avg_degree:.2f}")
    out(f"  density={density:.6f}")
    out(f"  feature_dim={feat.shape[1]}")
    out(f"  classes={int(labels.max()) + 1}")
    out(f"  train_size={len(train_idx)}")
    out(f"  test_size={len(test_idx)}")

    # 类别分布
    bins = np.bincount(labels[train_idx])
    out(f"\n[CLASS] 训练集类别分布")
    for cls, cnt in enumerate(bins):
        out(f"  Class{cls}={cnt}")
    imbalance = bins.max() / bins.max()
    out(f"  imbalance_ratio={bins.max() / bins.min():.1f}x")

    # 连通性分析
    n_components, labels_cc = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels_cc)
    largest_ratio = comp_sizes.max() / n_nodes if n_nodes > 0 else 0.0

    out(f"\n[GRAPH] 连通性分析")
    out(f"  connected_components={n_components}")
    out(f"  largest_component_ratio={largest_ratio:.2f}")
    out(f"  isolated_nodes={int((comp_sizes == 1).sum())}")
    out(f"  isolated_ratio={(comp_sizes == 1).sum() / n_nodes:.1%}")

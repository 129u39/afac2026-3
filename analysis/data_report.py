"""数据观测报告：输出数据集全景统计。"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def generate_report(data: dict, logger=None) -> dict:
    """生成完整数据报告并打印。

    Args:
        data: load_classification() 返回的数据字典
        logger: 打印函数，默认 print

    Returns:
        包含统计信息的字典
    """
    out = print if logger is None else logger

    adj = data["adj_csr"]
    feat = data["features"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    n_nodes = adj.shape[0]
    n_edges = adj.nnz
    avg_degree = n_edges / max(n_nodes, 1)
    density = n_edges / (n_nodes * n_nodes) if n_nodes else 0.0

    # 类别分布
    train_labels = labels[train_idx]
    bins = np.bincount(train_labels.astype(int))
    imbalance = float(bins.max() / max(bins.min(), 1))

    # 连通性
    n_comp, comp_labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(comp_labels)
    largest_ratio = float(comp_sizes.max() / max(n_nodes, 1))
    isolated = int((comp_sizes == 1).sum())
    isolated_ratio = isolated / max(n_nodes, 1)

    out("=" * 48)
    out("[DATA] 数据报告")
    out("=" * 48)
    out(f"  nodes={n_nodes}")
    out(f"  features={feat.shape[1]}")
    out(f"  classes={int(labels.max()) + 1}")

    out(f"\n[GRAPH]")
    out(f"  edges={n_edges}")
    out(f"  avg_degree={avg_degree:.2f}")
    out(f"  density={density:.6f}")
    out(f"  connected_components={n_comp}")
    out(f"  largest_component_ratio={largest_ratio:.2f}")
    out(f"  isolated_ratio={isolated_ratio:.1%}")

    out(f"\n[CLASS]")
    for cls, cnt in enumerate(bins):
        out(f"  class{cls}={cnt}")
    out(f"  imbalance_ratio={imbalance:.1f}")

    return {
        "nodes": n_nodes,
        "features": feat.shape[1],
        "classes": int(labels.max()) + 1,
        "edges": n_edges,
        "avg_degree": avg_degree,
        "density": density,
        "connected_components": n_comp,
        "largest_component_ratio": largest_ratio,
        "isolated_count": isolated,
        "isolated_ratio": isolated_ratio,
        "class_distribution": bins.tolist(),
        "imbalance_ratio": imbalance,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }

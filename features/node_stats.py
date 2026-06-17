"""节点统计特征：degree、is_isolated。

核心问题：31% 节点 degree=0，需要专门标记。
"""
import numpy as np


def compute_node_stats(adj_csr, features_dense=None):
    """计算节点统计特征并拼接。

    Args:
        adj_csr: CSR 邻接矩阵 (N, N)
        features_dense: 原始特征矩阵 (N, D)，可选

    Returns:
        combined: 拼接后的特征矩阵
    """
    deg = np.asarray(adj_csr.sum(axis=1)).ravel().astype(np.float32)
    is_isolated = (deg == 0).astype(np.float32).reshape(-1, 1)

    isolated_count = int(is_isolated.sum())
    total = len(deg)
    ratio = isolated_count / max(total, 1)

    print(f"[ISOLATED] count={isolated_count} ratio={ratio:.0%}")

    parts = []
    if features_dense is not None:
        parts.append(features_dense)
    parts.append(deg.reshape(-1, 1))      # degree
    parts.append(is_isolated)              # is_isolated flag

    combined = np.concatenate(parts, axis=1)
    return combined

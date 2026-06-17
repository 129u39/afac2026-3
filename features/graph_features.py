"""图结构特征：Degree、PageRank、KCore。

为节点特征矩阵添加图结构衍生的特征，增强模型对图拓扑的感知。
"""
import numpy as np
from scipy.sparse import csgraph


def compute_graph_features(adj_csr, features_dense=None):
    """计算图结构特征并拼接。

    Args:
        adj_csr: CSR 邻接矩阵 (N, N)
        features_dense: 可选的原始特征矩阵 (N, D)

    Returns:
        combined: 拼接后的特征矩阵
        feat_log: 特征构建日志字典
    """
    N = adj_csr.shape[0]
    feat_list = []
    log = {}

    # 1. Degree + Log-Degree
    deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float32)
    log_deg = np.log1p(deg).reshape(-1, 1)
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-10)
    feat_list.append(deg_norm.reshape(-1, 1))
    feat_list.append(log_deg)
    log["degree"] = True
    log["log_degree"] = True

    # 2. PageRank
    try:
        pagerank = csgraph.pagerank(adj_csr.astype(np.float32))
        feat_list.append(pagerank.reshape(-1, 1))
        log["pagerank"] = True
    except Exception:
        log["pagerank"] = False

    # 3. KCore
    try:
        kcores = csgraph.structural_rank(adj_csr)
        feat_list.append(kcores.reshape(-1, 1))
        log["kcore"] = True
    except Exception:
        log["kcore"] = False

    # 4. 原始特征（如果有）
    if features_dense is not None:
        feat_list.insert(0, features_dense)
        log["raw_features"] = True

    combined = np.concatenate(feat_list, axis=1)
    log["new_dim"] = combined.shape[1]

    print(f"[GRAPH_FEATURE] degree={log.get('degree', False)} "
          f"pagerank={log.get('pagerank', False)} "
          f"kcore={log.get('kcore', False)} "
          f"new_dim={combined.shape[1]}")

    return combined, log

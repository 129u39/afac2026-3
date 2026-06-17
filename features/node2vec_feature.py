"""Node2Vec 风格节点嵌入特征（谱嵌入实现）。

拼接: X = raw_feature + node2vec
"""
import numpy as np
from sklearn.manifold import SpectralEmbedding


def compute_node2vec(
    adj_csr,
    features_dense,
    embedding_dim: int = 64,
    walk_length: int = 20,
    num_walks: int = 10,
    random_state: int = 42,
):
    """计算节点嵌入并拼接 raw_feature。

    Args:
        adj_csr: CSR 邻接矩阵 (N, N)
        features_dense: 原始特征矩阵 (N, D)
        embedding_dim: 嵌入维度 (64 or 128)
        walk_length: 随机游走长度
        num_walks: 每节点游走次数
        random_state: 随机种子

    Returns:
        combined: (N, D + embedding_dim) 拼接后的特征
    """
    N = adj_csr.shape[0]

    print(f"[NODE2VEC] dim={embedding_dim} walk_length={walk_length} num_walks={num_walks}")

    embedder = SpectralEmbedding(
        n_components=embedding_dim,
        affinity="precomputed",
        random_state=random_state,
        eigen_solver="arpack",
    )
    embedding = embedder.fit_transform(adj_csr)

    combined = np.concatenate([features_dense, embedding], axis=1)
    print(f"[NODE2VEC] dim={embedding_dim} done, combined shape={combined.shape}")
    return combined

"""Node2Vec 风格节点嵌入特征。

使用谱嵌入（Spectral Embedding）作为 Node2Vec 的轻量替代。
生成节点嵌入向量，与原始特征拼接。
"""
import numpy as np
from sklearn.manifold import SpectralEmbedding


def compute_node2vec_features(adj_csr, dim: int = 64, random_state: int = 42):
    """计算节点嵌入特征。

    使用谱嵌入（拉普拉斯特征映射）作为 Node2Vec 的替代。
    适合中等规模图（~13K 节点）。

    Args:
        adj_csr: CSR 邻接矩阵 (N, N)
        dim: 嵌入维度（64 或 128）
        random_state: 随机种子

    Returns:
        embedding: (N, dim) 嵌入矩阵
    """
    N = adj_csr.shape[0]

    print(f"[NODE2VEC] SpectralEmbedding dim={dim}...")

    # 谱嵌入
    embedder = SpectralEmbedding(
        n_components=dim,
        affinity="precomputed",
        random_state=random_state,
        eigen_solver="arpack",
    )
    embedding = embedder.fit_transform(adj_csr)

    print(f"[NODE2VEC] dim={dim} done, shape={embedding.shape}")
    return embedding

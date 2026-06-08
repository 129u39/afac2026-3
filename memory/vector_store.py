"""向量存储：将实验配置编码为向量，支持相似度检索。"""

import numpy as np
from typing import Any


# 模型类型编码映射
CLS_MODEL_ENCODING = {"GCN": 0, "GAT": 1, "GraphSAGE": 2}
REC_MODEL_ENCODING = {"Popularity": 0, "ItemCF": 1, "BPR_MF": 2, "SASRec": 3, "LightGCN": 4}


class ConfigEncoder:
    """将实验配置编码为固定长度的数值向量。"""

    def __init__(self, task_type: str):
        """
        Args:
            task_type: "classification" 或 "recommendation"
        """
        self.task_type = task_type
        if task_type == "classification":
            self.model_encoding = CLS_MODEL_ENCODING
            self.feature_names = [
                "model_type", "hidden_dim", "num_layers", "dropout", "lr", "weight_decay"
            ]
        else:
            self.model_encoding = REC_MODEL_ENCODING
            self.feature_names = [
                "model_type", "embedding_dim", "lr", "batch_size", "weight_decay"
            ]

    def encode(self, config: dict) -> np.ndarray:
        """将配置编码为向量。

        Args:
            config: 实验配置字典

        返回:
            归一化的数值向量
        """
        if self.task_type == "classification":
            return self._encode_cls(config)
        else:
            return self._encode_rec(config)

    def _encode_cls(self, config: dict) -> np.ndarray:
        """分类配置编码。"""
        model_type = config.get("model_type", "GCN")
        model_idx = self.model_encoding.get(model_type, 0) / max(len(self.model_encoding) - 1, 1)

        hidden_dim = config.get("hidden_dim", 64) / 256.0
        num_layers = config.get("num_layers", 2) / 3.0
        dropout = config.get("dropout", 0.5)
        lr = config.get("lr", 0.01) * 100  # 归一化到 0~1
        weight_decay = config.get("weight_decay", 5e-4) * 1000

        return np.array([model_idx, hidden_dim, num_layers, dropout, lr, weight_decay], dtype=np.float32)

    def _encode_rec(self, config: dict) -> np.ndarray:
        """推荐配置编码。"""
        model_type = config.get("model_type", "BPR_MF")
        model_idx = self.model_encoding.get(model_type, 0) / max(len(self.model_encoding) - 1, 1)

        embedding_dim = config.get("embedding_dim", 64) / 128.0
        lr = config.get("lr", 0.01) * 100
        batch_size = config.get("batch_size", 256) / 512.0
        weight_decay = config.get("weight_decay", 1e-5) * 10000

        return np.array([model_idx, embedding_dim, lr, batch_size, weight_decay], dtype=np.float32)


class VectorStore:
    """向量存储与检索。"""

    def __init__(self):
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict] = []

    def add(self, vector: np.ndarray, metadata: dict):
        """添加向量及其元数据。

        Args:
            vector: 编码后的向量
            metadata: 关联的实验元数据
        """
        self.vectors.append(vector)
        self.metadata.append(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """检索最相似的 k 个向量。

        使用余弦相似度。

        Args:
            query_vector: 查询向量
            k: 返回数量

        返回:
            按相似度排序的元数据列表，每个附带 "similarity" 字段
        """
        if not self.vectors:
            return []

        # 计算余弦相似度
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = self._cosine_similarity(query_vector, vec)
            similarities.append((sim, i))

        # 排序并返回 top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, idx in similarities[:k]:
            meta = self.metadata[idx].copy()
            meta["similarity"] = sim
            results.append(meta)

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def __len__(self):
        return len(self.vectors)

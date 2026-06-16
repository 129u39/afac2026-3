"""检索器：基于向量相似度检索历史实验。"""

from memory.vector_store import ConfigEncoder, VectorStore
from memory import ExperimentMemory, ExperimentRecord


class Retriever:
    """实验检索器：根据配置相似度检索历史实验。

    用途：
    - 在规划新实验时，找到最相似的历史实验作为参考
    - 避免重复尝试相似的配置
    - 利用历史经验指导新实验
    """

    def __init__(self, task_type: str):
        """
        Args:
            task_type: "classification" 或 "recommendation"
        """
        self.task_type = task_type
        self.encoder = ConfigEncoder(task_type)
        self.store = VectorStore()

    def index(self, memory: ExperimentMemory):
        """将实验记忆中的所有记录索引到向量存储。

        Args:
            memory: 实验记忆
        """
        self.store = VectorStore()
        for rec in memory.records:
            vector = self.encoder.encode(rec.config)
            metadata = {
                "exp_id": rec.exp_id,
                "round_num": rec.round_num,
                "model_type": rec.model_type,
                "config": rec.config,
                "metrics": rec.metrics,
                "elapsed_seconds": rec.elapsed_seconds,
            }
            self.store.add(vector, metadata)

    def top_k_similar(self, config: dict, k: int = 5) -> list[dict]:
        """检索与给定配置最相似的历史实验。

        Args:
            config: 实验配置
            k: 返回数量

        返回:
            按相似度排序的实验列表，每个包含 config, metrics, similarity
        """
        query_vector = self.encoder.encode(config)
        return self.store.search(query_vector, k=k)

    def find_most_similar(self, config: dict) -> dict | None:
        """找到最相似的历史实验。

        Args:
            config: 实验配置

        返回:
            最相似的实验元数据，或 None
        """
        results = self.top_k_similar(config, k=1)
        return results[0] if results else None

    def is_too_similar(self, config: dict, threshold: float = 0.85) -> bool:
        """检查配置是否与已有实验过于相似。

        Args:
            config: 实验配置
            threshold: 相似度阈值

        返回:
            如果最相似的实验相似度 >= threshold 则返回 True
        """
        result = self.find_most_similar(config)
        if result is None:
            return False
        return result.get("similarity", 0) >= threshold

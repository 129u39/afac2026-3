"""实验记忆模块：记录、查询、排序历史实验。"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExperimentRecord:
    """单轮实验记录。"""
    round_num: int
    model_type: str
    config: dict[str, Any]
    metrics: dict[str, float]
    elapsed_seconds: float
    notes: str = ""


class ExperimentMemory:
    """实验记忆管理器。"""

    def __init__(self):
        self.records: list[ExperimentRecord] = []
        self._best_cls: ExperimentRecord | None = None
        self._best_rec: ExperimentRecord | None = None

    def record(self, rec: ExperimentRecord):
        """添加一条实验记录。"""
        self.records.append(rec)

        # 更新最佳分类实验
        if "val_accuracy" in rec.metrics:
            if self._best_cls is None or rec.metrics["val_accuracy"] > self._best_cls.metrics.get("val_accuracy", 0):
                self._best_cls = rec

        # 更新最佳推荐实验
        if "ndcg@k" in rec.metrics:
            if self._best_rec is None or rec.metrics["ndcg@k"] > self._best_rec.metrics.get("ndcg@k", 0):
                self._best_rec = rec

    def best_classification(self) -> ExperimentRecord | None:
        """返回最佳分类实验。"""
        return self._best_cls

    def best_recommendation(self) -> ExperimentRecord | None:
        """返回最佳推荐实验。"""
        return self._best_rec

    def by_model_type(self, model_type: str) -> list[ExperimentRecord]:
        """按模型类型筛选实验。"""
        return [r for r in self.records if r.model_type == model_type]

    def recent(self, n: int = 5) -> list[ExperimentRecord]:
        """返回最近 n 条实验。"""
        return self.records[-n:]

    def all_sorted_by_metric(self, metric_key: str, reverse: bool = True) -> list[ExperimentRecord]:
        """按指定指标排序所有实验。"""
        return sorted(
            self.records,
            key=lambda r: r.metrics.get(metric_key, 0),
            reverse=reverse,
        )

    def to_json(self) -> str:
        """序列化为 JSON。"""
        return json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """生成人类可读的实验摘要。"""
        lines = [f"共 {len(self.records)} 轮实验"]
        if self._best_cls:
            lines.append(f"最佳分类: round={self._best_cls.round_num}, "
                         f"model={self._best_cls.model_type}, "
                         f"acc={self._best_cls.metrics.get('val_accuracy', 0):.4f}")
        if self._best_rec:
            lines.append(f"最佳推荐: round={self._best_rec.round_num}, "
                         f"model={self._best_rec.model_type}, "
                         f"ndcg={self._best_rec.metrics.get('ndcg@k', 0):.4f}")
        return "\n".join(lines)

"""实验记忆模块：记录、查询、排序历史实验。V1 — 增强持久化与检索。"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class ExperimentRecord:
    """单轮实验记录。"""
    round_num: int
    model_type: str
    config: dict[str, Any]
    metrics: dict[str, float]
    elapsed_seconds: float
    # V1 新增字段
    exp_id: str = ""
    task: str = ""           # "classification" | "recommendation"
    timestamp: str = ""
    status: str = "success"  # "success" | "failed" | "timeout"
    notes: str = ""

    def __post_init__(self):
        if not self.exp_id:
            self.exp_id = uuid.uuid4().hex[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExperimentMemory:
    """实验记忆管理器。支持持久化与多维检索。"""

    def __init__(self):
        self.records: list[ExperimentRecord] = []
        self._best_cls: ExperimentRecord | None = None
        self._best_rec: ExperimentRecord | None = None

    # ── 基础操作 ──────────────────────────────────────

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

    def add(self, rec: ExperimentRecord):
        """record() 的别名，兼容 V1 API。"""
        self.record(rec)

    # ── 查询 API ──────────────────────────────────────

    def best_classification(self) -> ExperimentRecord | None:
        """返回最佳分类实验。"""
        return self._best_cls

    def best_recommendation(self) -> ExperimentRecord | None:
        """返回最佳推荐实验。"""
        return self._best_rec

    def get_best(self, task: str) -> ExperimentRecord | None:
        """返回指定任务的最佳实验。"""
        if task == "classification":
            return self._best_cls
        elif task == "recommendation":
            return self._best_rec
        return None

    def get_last_k(self, k: int = 5) -> list[ExperimentRecord]:
        """返回最近 k 条实验记录。"""
        return self.records[-k:]

    def get_by_model(self, model_type: str) -> list[ExperimentRecord]:
        """按模型类型筛选实验。"""
        return [r for r in self.records if r.model_type == model_type]

    def by_model_type(self, model_type: str) -> list[ExperimentRecord]:
        """get_by_model 的别名，兼容旧 API。"""
        return self.get_by_model(model_type)

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

    # ── 持久化 ────────────────────────────────────────

    def save(self, path: str):
        """保存实验记录到 JSON 文件。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "records": [asdict(r) for r in self.records],
            "best_cls_id": self._best_cls.exp_id if self._best_cls else None,
            "best_rec_id": self._best_rec.exp_id if self._best_rec else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从 JSON 文件加载实验记录。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.records = []
        self._best_cls = None
        self._best_rec = None

        for r_dict in data.get("records", []):
            rec = ExperimentRecord(**r_dict)
            self.records.append(rec)

        # 恢复最佳记录引用
        best_cls_id = data.get("best_cls_id")
        best_rec_id = data.get("best_rec_id")
        for r in self.records:
            if best_cls_id and r.exp_id == best_cls_id:
                self._best_cls = r
            if best_rec_id and r.exp_id == best_rec_id:
                self._best_rec = r

        # 如果 ID 匹配失败，重新计算最佳
        if self._best_cls is None:
            for r in self.records:
                if "val_accuracy" in r.metrics:
                    if self._best_cls is None or r.metrics["val_accuracy"] > self._best_cls.metrics.get("val_accuracy", 0):
                        self._best_cls = r
        if self._best_rec is None:
            for r in self.records:
                if "ndcg@k" in r.metrics:
                    if self._best_rec is None or r.metrics["ndcg@k"] > self._best_rec.metrics.get("ndcg@k", 0):
                        self._best_rec = r

    # ── 序列化与摘要 ──────────────────────────────────

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

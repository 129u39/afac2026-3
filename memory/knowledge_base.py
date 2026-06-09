"""知识库：跨任务经验存储与检索。"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class KnowledgeEntry:
    """单条知识记录。"""
    model_name: str
    task_type: str
    best_metric: float
    best_config: dict = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    timestamp: str = ""


class KnowledgeBase:
    """跨任务知识库。

    记录模型在不同任务上的表现，支持知识迁移。
    例如：GraphSAGE 在分类任务上表现好，其经验可迁移到推荐任务。
    """

    def __init__(self, path: str = "output/knowledge_base.json"):
        """
        Args:
            path: 知识库持久化路径
        """
        self.path = path
        self.entries: list[KnowledgeEntry] = []
        self._load()

    def add(self, entry: KnowledgeEntry):
        """添加知识条目。"""
        # 检查是否已有相同记录，有则更新
        for existing in self.entries:
            if existing.model_name == entry.model_name and existing.task_type == entry.task_type:
                if entry.best_metric > existing.best_metric:
                    existing.best_metric = entry.best_metric
                    existing.best_config = entry.best_config
                    existing.insights = entry.insights
                    existing.timestamp = entry.timestamp
                return
        self.entries.append(entry)

    def get_relevant_knowledge(
        self,
        task_type: str,
        model_name: str | None = None,
    ) -> list[KnowledgeEntry]:
        """获取相关知识。

        Args:
            task_type: 当前任务类型
            model_name: 当前模型名称（可选）

        返回:
            相关知识条目列表
        """
        results = []
        for entry in self.entries:
            # 同任务的知识
            if entry.task_type == task_type:
                results.append(entry)
            # 同模型跨任务的知识（知识迁移）
            elif model_name and entry.model_name == model_name:
                results.append(entry)
        return sorted(results, key=lambda e: e.best_metric, reverse=True)

    def get_model_summary(self) -> dict:
        """获取模型表现摘要。"""
        summary: dict[str, dict] = {}
        for entry in self.entries:
            if entry.model_name not in summary:
                summary[entry.model_name] = {}
            summary[entry.model_name][entry.task_type] = {
                "best_metric": entry.best_metric,
                "insights": entry.insights,
            }
        return summary

    def format_for_prompt(self, task_type: str, model_name: str | None = None) -> str:
        """格式化知识为提示词文本。"""
        entries = self.get_relevant_knowledge(task_type, model_name)
        if not entries:
            return "暂无相关知识。"

        lines = []
        for entry in entries:
            lines.append(
                f"- {entry.model_name} ({entry.task_type}): "
                f"metric={entry.best_metric:.4f}, "
                f"insights={', '.join(entry.insights[:3])}"
            )
        return "\n".join(lines)

    def save(self):
        """保存知识库到文件。"""
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载知识库。"""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [KnowledgeEntry(**d) for d in data]
        except Exception:
            self.entries = []

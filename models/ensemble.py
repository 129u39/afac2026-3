"""Ensemble Builder：集成多个模型的预测。"""

import torch
import numpy as np
from typing import Any


class EnsembleBuilder:
    """集成构建器。

    收集 Top-K 模型，通过加权平均生成最终预测。

    分类：softmax 概率平均
    推荐：分数向量平均
    """

    def __init__(self, task_type: str):
        """
        Args:
            task_type: "classification" 或 "recommendation"
        """
        self.task_type = task_type
        self.models: list[dict] = []  # [{"model": ..., "config": ..., "metric": ..., "weight": ...}]

    def add_model(self, model: Any, config: dict, metric: float, weight: float | None = None):
        """添加模型到集成池。

        Args:
            model: 训练好的模型
            config: 模型配置
            metric: 评估指标
            weight: 权重（None 时自动计算）
        """
        self.models.append({
            "model": model,
            "config": config,
            "metric": metric,
            "weight": weight,
        })

    def build(self, weights: str = "metric"):
        """构建集成。

        Args:
            weights: 权重策略
                - "equal": 等权重
                - "metric": 按指标加权
                - "softmax": softmax 归一化
        """
        if not self.models:
            return

        if weights == "equal":
            for m in self.models:
                m["weight"] = 1.0 / len(self.models)
        elif weights == "metric":
            total = sum(m["metric"] for m in self.models)
            if total > 0:
                for m in self.models:
                    m["weight"] = m["metric"] / total
            else:
                for m in self.models:
                    m["weight"] = 1.0 / len(self.models)
        elif weights == "softmax":
            metrics = np.array([m["metric"] for m in self.models])
            exp_metrics = np.exp(metrics - np.max(metrics))
            softmax_weights = exp_metrics / exp_metrics.sum()
            for i, m in enumerate(self.models):
                m["weight"] = float(softmax_weights[i])

    def predict_cls(self, data, device: torch.device = None) -> np.ndarray:
        """分类集成预测。

        Args:
            data: PyG Data 对象
            device: 计算设备

        返回:
            预测标签数组
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if device is None:
            device = next(iter(self.models))["model"].parameters().device

        # 收集所有模型的 softmax 概率
        all_probs = []
        for m in self.models:
            model = m["model"]
            weight = m["weight"] or 1.0 / len(self.models)

            model.eval()
            with torch.no_grad():
                out = model(data)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs * weight)

        # 加权平均
        avg_probs = torch.stack(all_probs).sum(dim=0)
        predictions = avg_probs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_cls_softmax(self, data, device: torch.device = None) -> torch.Tensor:
        """分类集成预测（返回 softmax 概率）。

        用于更精细的集成或评估。
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if device is None:
            device = next(iter(self.models))["model"].parameters().device

        all_probs = []
        for m in self.models:
            model = m["model"]
            weight = m["weight"] or 1.0 / len(self.models)

            model.eval()
            with torch.no_grad():
                out = model(data)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs * weight)

        return torch.stack(all_probs).sum(dim=0)

    def predict_rec(self, uid: str, seq_dedup: list[str], top_k: int = 10) -> list[str]:
        """推荐集成预测。

        通过分数平均生成推荐列表。

        Args:
            uid: 用户 ID
            seq_dedup: 用户历史序列
            top_k: 返回数量

        返回:
            推荐物品列表
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # 收集所有模型的推荐分数
        all_scores: dict[str, float] = {}
        for m in self.models:
            model = m["model"]
            weight = m["weight"] or 1.0 / len(self.models)

            try:
                # 获取模型的推荐列表
                recs = model.predict(uid, seq_dedup, top_k=len(model.all_iid) if hasattr(model, 'all_iid') else 100)
                for i, item_id in enumerate(recs):
                    # 分数 = 位置倒数 × 权重
                    score = (len(recs) - i) / len(recs) * weight
                    all_scores[item_id] = all_scores.get(item_id, 0) + score
            except Exception:
                continue

        # 按分数排序
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # 排除已交互物品
        seq_set = set(seq_dedup)
        result = [item_id for item_id, _ in sorted_items if item_id not in seq_set]

        return result[:top_k]

    def __len__(self):
        return len(self.models)

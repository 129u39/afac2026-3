"""TopK Candidate Pool：维护最优配置池，指导搜索方向。"""

import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PoolEntry:
    """配置池条目。"""
    rank: int
    metric: float
    config: dict
    model_name: str
    train_time: float = 0.0
    round_num: int = 0


class TopKPool:
    """Top-K 配置池。

    维护最优的 K 个配置，后续搜索围绕这些配置展开。
    80% 搜索围绕 TopK，20% 随机探索。
    """

    def __init__(self, max_size: int = 20, path: str = "output/top_pool.json"):
        """
        Args:
            max_size: 池最大容量
            path: 持久化路径
        """
        self.max_size = max_size
        self.path = path
        self.entries: list[PoolEntry] = []
        self._load()

    def add(self, config: dict, metric: float, model_name: str = "", train_time: float = 0.0, round_num: int = 0):
        """添加配置到池中。

        Args:
            config: 实验配置
            metric: 评估指标（越大越好）
            model_name: 模型名称
            train_time: 训练时间
            round_num: 轮次号
        """
        # 检查是否已存在相同配置
        for entry in self.entries:
            if entry.config == config:
                if metric > entry.metric:
                    entry.metric = metric
                    entry.train_time = train_time
                    entry.round_num = round_num
                self._re_rank()
                return

        # 添加新条目
        entry = PoolEntry(
            rank=0,
            metric=metric,
            config=config,
            model_name=model_name,
            train_time=train_time,
            round_num=round_num,
        )
        self.entries.append(entry)
        self._re_rank()

        # 保留 Top-K
        if len(self.entries) > self.max_size:
            self.entries = self.entries[:self.max_size]

    def get_top_k(self, k: int = 5) -> list[dict]:
        """返回 Top-K 配置。"""
        return [e.config for e in self.entries[:k]]

    def get_best(self) -> dict | None:
        """返回最佳配置。"""
        if not self.entries:
            return None
        return self.entries[0].config

    def get_best_metric(self) -> float:
        """返回最佳指标。"""
        if not self.entries:
            return 0.0
        return self.entries[0].metric

    def get_focus_configs(self, n: int = 10, focus_ratio: float = 0.8) -> list[dict]:
        """生成聚焦配置。

        Args:
            n: 生成配置数量
            focus_ratio: 围绕 TopK 的比例

        返回:
            配置列表
        """
        configs = []
        n_focus = int(n * focus_ratio)
        n_explore = n - n_focus

        # 围绕 TopK 生成（微调）
        top_configs = self.get_top_k(min(5, len(self.entries)))
        for _ in range(n_focus):
            if top_configs:
                base = random.choice(top_configs)
                configs.append(self._perturb(base))

        # 随机探索
        for _ in range(n_explore):
            configs.append(self._random_config())

        return configs

    def _re_rank(self):
        """重新排序。"""
        self.entries.sort(key=lambda e: e.metric, reverse=True)
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1

    def _perturb(self, config: dict) -> dict:
        """微调配置。"""
        import copy
        cfg = copy.deepcopy(config)
        param = random.choice(list(cfg.keys()))
        if param == "hidden_dim":
            cfg["hidden_dim"] = random.choice([64, 128, 256, 512])
        elif param == "num_layers":
            cfg["num_layers"] = random.choice([2, 3, 4])
        elif param == "dropout":
            cfg["dropout"] = random.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
        elif param == "lr":
            cfg["lr"] = random.choice([5e-4, 1e-3, 2e-3, 3e-3, 5e-3])
        elif param == "weight_decay":
            cfg["weight_decay"] = random.choice([0.0, 1e-5, 5e-5, 1e-4, 5e-4])
        return cfg

    def _random_config(self) -> dict:
        """生成随机配置。"""
        return {
            "model_type": random.choice(["GCN", "GAT", "GraphSAGE"]),
            "hidden_dim": random.choice([64, 128, 256, 512]),
            "num_layers": random.choice([2, 3, 4]),
            "dropout": random.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5]),
            "lr": random.choice([5e-4, 1e-3, 2e-3, 3e-3, 5e-3]),
            "weight_decay": random.choice([0.0, 1e-5, 5e-5, 1e-4, 5e-4]),
            "epochs": 200,
            "patience": 30,
        }

    def save(self):
        """保存到文件。"""
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载。"""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [PoolEntry(**d) for d in data]
        except Exception:
            self.entries = []

    def summary(self) -> str:
        """生成摘要。"""
        if not self.entries:
            return "TopK Pool: 空"
        lines = [f"TopK Pool: {len(self.entries)} 条目"]
        for e in self.entries[:5]:
            lines.append(f"  #{e.rank}: {e.model_name} metric={e.metric:.4f}")
        return "\n".join(lines)

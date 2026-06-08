"""策略规划器：基于实验记忆和反馈，决定下一轮实验配置。"""

import copy
import itertools
import random

from memory import ExperimentMemory
from feedback_analyzer import FeedbackAnalyzer


class Planner:
    """实验策略规划器。"""

    def __init__(self, task_type: str, seed: int = 42):
        self.task_type = task_type
        self.analyzer = FeedbackAnalyzer()
        self.rng = random.Random(seed)
        self._search_iter = None
        self._tried_configs: list[dict] = []

    def next_config(self, memory: ExperimentMemory) -> dict:
        """根据实验记忆生成下一轮配置。

        返回:
            config dict，包含 model_type 和所有超参
        """
        feedback = self.analyzer.analyze(memory, self.task_type)

        if not memory.records:
            return self._initial_config()

        best = feedback["best_config"]
        trend = feedback["trend"]
        tried_models = feedback["tried_models"]

        if self.task_type == "classification":
            return self._next_cls_config(memory, best, trend, tried_models)
        else:
            return self._next_rec_config(memory, best, trend, tried_models)

    def _initial_config(self) -> dict:
        """初始配置：快速基线。"""
        if self.task_type == "classification":
            return {
                "model_type": "GCN",
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.5,
                "lr": 0.01,
                "weight_decay": 5e-4,
                "epochs": 200,
                "patience": 30,
            }
        else:
            return {
                "model_type": "Popularity",
            }

    def _next_cls_config(self, memory, best, trend, tried_models) -> dict:
        """分类任务下一轮配置。"""
        all_models = ["GCN", "GAT", "GraphSAGE"]
        untried = [m for m in all_models if m not in tried_models]

        # 策略1：尝试未试过的模型
        if untried and self.rng.random() < 0.6:
            model = self.rng.choice(untried)
            return self._random_cls_config(model)

        # 策略2：在最佳模型上微调
        if best:
            return self._perturb_cls_config(best)

        # 策略3：随机探索
        model = self.rng.choice(all_models)
        return self._random_cls_config(model)

    def _random_cls_config(self, model_type: str) -> dict:
        """为指定模型生成随机配置。"""
        return {
            "model_type": model_type,
            "hidden_dim": self.rng.choice([64, 128, 256]),
            "num_layers": self.rng.choice([2, 3]),
            "dropout": self.rng.choice([0.0, 0.3, 0.5]),
            "lr": self.rng.choice([0.001, 0.005, 0.01]),
            "weight_decay": self.rng.choice([0.0, 5e-4]),
            "epochs": 200,
            "patience": 30,
        }

    def _perturb_cls_config(self, best_config: dict) -> dict:
        """在最佳配置基础上微调。"""
        config = copy.deepcopy(best_config)
        # 随机修改一个超参
        param = self.rng.choice(["hidden_dim", "dropout", "lr", "weight_decay", "num_layers"])

        if param == "hidden_dim":
            config["hidden_dim"] = self.rng.choice([64, 128, 256])
        elif param == "dropout":
            config["dropout"] = self.rng.choice([0.0, 0.2, 0.3, 0.5])
        elif param == "lr":
            config["lr"] = self.rng.choice([0.0005, 0.001, 0.005, 0.01])
        elif param == "weight_decay":
            config["weight_decay"] = self.rng.choice([0.0, 1e-5, 5e-4, 1e-3])
        elif param == "num_layers":
            config["num_layers"] = self.rng.choice([2, 3])

        return config

    def _next_rec_config(self, memory, best, trend, tried_models) -> dict:
        """推荐任务下一轮配置。"""
        all_models = ["Popularity", "ItemCF", "BPR_MF", "SASRec"]
        untried = [m for m in all_models if m not in tried_models]

        # 策略：按复杂度递增尝试
        if untried:
            # 优先尝试更复杂的模型
            priority = ["ItemCF", "BPR_MF", "SASRec"]
            for m in priority:
                if m in untried:
                    return self._default_rec_config(m)
            return self._default_rec_config(untried[0])

        # 所有模型都试过了，在最佳上微调
        if best:
            return self._perturb_rec_config(best)

        return self._default_rec_config("BPR_MF")

    def _default_rec_config(self, model_type: str) -> dict:
        """推荐模型默认配置。"""
        base = {"model_type": model_type}
        if model_type == "BPR_MF":
            base.update({"embedding_dim": 64, "lr": 0.01, "epochs": 50, "batch_size": 512, "weight_decay": 1e-5})
        elif model_type == "SASRec":
            base.update({"embedding_dim": 64, "lr": 0.001, "epochs": 50, "batch_size": 256, "weight_decay": 1e-5})
        return base

    def _perturb_rec_config(self, best_config: dict) -> dict:
        """在最佳推荐配置基础上微调。"""
        config = copy.deepcopy(best_config)
        if "embedding_dim" in config:
            config["embedding_dim"] = self.rng.choice([32, 64, 128])
        if "lr" in config:
            config["lr"] = self.rng.choice([0.0005, 0.001, 0.005, 0.01])
        return config

    def should_stop(self, memory: ExperimentMemory, max_no_improve: int = 5) -> bool:
        """判断是否应该停止实验（连续多轮无提升）。"""
        if len(memory.records) < 3:
            return False

        metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"
        recent = memory.recent(max_no_improve + 1)
        if len(recent) < max_no_improve + 1:
            return False

        best_recent = max(r.metrics.get(metric_key, 0) for r in recent)
        first_metric = recent[0].metrics.get(metric_key, 0)

        # 如果最近几轮没有超过第一轮的指标，停止
        no_improve_count = 0
        current_best = first_metric
        for r in recent[1:]:
            if r.metrics.get(metric_key, 0) > current_best + 1e-6:
                current_best = r.metrics.get(metric_key, 0)
                no_improve_count = 0
            else:
                no_improve_count += 1

        return no_improve_count >= max_no_improve

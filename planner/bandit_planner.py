"""Bandit Planner：使用 UCB 策略选择模型架构，结合微调逻辑生成配置。"""

import copy
import random

from planner.bandit import UCBBandit
from memory import ExperimentMemory
from feedback_analyzer import FeedbackAnalyzer


# 分类任务默认配置模板
CLS_DEFAULTS = {
    # 主力模型：LightGBM（特征工程）
    "LightGBM": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
    # 主力模型：MLP
    "MLP": {"hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 30},
    # 图模型：GraphSAGE
    "GraphSAGE": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2, "lr": 0.005, "weight_decay": 5e-4, "epochs": 200, "patience": 30},
    # 图模型：GCNII
    "GCNII": {"hidden_dim": 256, "num_layers": 16, "dropout": 0.3, "lr": 0.01, "weight_decay": 5e-4, "alpha": 0.1, "theta": 0.5, "epochs": 200, "patience": 30},
}

# 推荐任务默认配置模板
REC_DEFAULTS = {
    "Popularity": {},
    "ItemCF": {},
    "BPR_MF": {"embedding_dim": 64, "lr": 0.01, "epochs": 50, "batch_size": 512, "weight_decay": 1e-5},
    "SASRec": {"embedding_dim": 64, "lr": 0.001, "epochs": 50, "batch_size": 256, "weight_decay": 1e-5},
    "LightGCN": {"embedding_dim": 64, "num_layers": 3, "lr": 0.001, "epochs": 50, "batch_size": 512, "weight_decay": 1e-5},
}


class BanditPlanner:
    """基于 UCB Bandit 的实验规划器。

    职责：
    1. 使用 UCB 选择模型架构（exploration vs exploitation）
    2. 在选定模型的基础上生成超参配置
    3. 根据反馈微调配置
    """

    def __init__(self, task_type: str, seed: int = 42, c: float = 1.5):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            seed: 随机种子
            c: UCB 探索系数
        """
        self.task_type = task_type
        self.analyzer = FeedbackAnalyzer()
        self.rng = random.Random(seed)

        # 根据任务类型初始化 Bandit 臂
        if task_type == "classification":
            arms = ["LightGBM", "MLP", "GraphSAGE", "GCNII"]
        else:
            arms = ["LightGCN", "SASRec", "BPR_MF", "ItemCF"]

        self.bandit = UCBBandit(arms=arms, c=c)
        self._last_selected_arm: str | None = None

    def next_config(self, memory: ExperimentMemory) -> dict:
        """根据 Bandit 选择和实验记忆生成下一轮配置。

        流程:
        1. 用 UCB 选择模型架构
        2. 如果该模型有历史最佳配置，在此基础上微调
        3. 否则使用默认配置

        注意: 调用前应先调用 _update_bandit_from_memory() 更新统计。

        返回:
            config dict，包含 model_type 和所有超参
        """
        feedback = self.analyzer.analyze(memory, self.task_type)

        # 用 UCB 选择下一轮模型
        arm = self.bandit.select_arm()
        self._last_selected_arm = arm

        # 检查该模型是否有历史记录
        model_records = memory.get_by_model(arm)
        if model_records:
            # 找到该模型的最佳配置，微调
            metric_key = feedback["metric_key"]
            best_for_model = max(model_records, key=lambda r: r.metrics.get(metric_key, 0))
            return self._perturb_config(best_for_model.config, arm)
        else:
            # 该模型没有历史记录，使用默认配置
            return self._default_config(arm)

    def get_last_decision(self) -> dict:
        """返回上一次决策的详情（用于日志记录）。"""
        return {
            "bandit_arm": self._last_selected_arm,
            "bandit_stats": self.bandit.get_stats(),
        }

    def _update_bandit_from_memory(self, memory: ExperimentMemory):
        """从实验记忆中更新 Bandit 统计。

        用总记录数与 bandit 总拉杆数的差值判断是否有未更新的记录。
        """
        if not memory.records:
            return

        metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"

        # 检查有多少记录还没被更新到 bandit
        total_records = len(memory.records)
        total_pulls = self.bandit.total_pulls

        if total_records <= total_pulls:
            return  # 所有记录都已更新

        # 更新未处理的记录（从 total_pulls 开始）
        for i in range(total_pulls, total_records):
            rec = memory.records[i]
            reward = rec.metrics.get(metric_key, 0.0)
            arm = rec.model_type
            if arm in self.bandit.arms:
                self.bandit.update(arm, reward)

    def _default_config(self, model_type: str) -> dict:
        """生成模型的默认配置。"""
        if self.task_type == "classification":
            base = CLS_DEFAULTS.get(model_type, CLS_DEFAULTS["GCNII"]).copy()
            base["model_type"] = model_type
            return base
        else:
            base = REC_DEFAULTS.get(model_type, {}).copy()
            base["model_type"] = model_type
            return base

    def _perturb_config(self, best_config: dict, model_type: str) -> dict:
        """在最佳配置基础上微调。"""
        config = copy.deepcopy(best_config)
        config["model_type"] = model_type

        if self.task_type == "classification":
            # 通用参数
            params = ["hidden_dim", "dropout", "lr", "weight_decay", "num_layers"]

            # 添加模型特有参数
            if model_type == "APPNP":
                params.extend(["K", "alpha"])
            elif model_type == "GCNII":
                params.extend(["alpha", "theta"])

            param = self.rng.choice(params)

            if param == "hidden_dim":
                config["hidden_dim"] = self.rng.choice([64, 128, 256, 512])
            elif param == "dropout":
                config["dropout"] = self.rng.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
            elif param == "lr":
                config["lr"] = self.rng.choice([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
            elif param == "weight_decay":
                config["weight_decay"] = self.rng.choice([0.0, 1e-5, 5e-5, 1e-4, 5e-4])
            elif param == "num_layers":
                if model_type == "GCNII":
                    config["num_layers"] = self.rng.choice([4, 8, 12, 16])
                else:
                    config["num_layers"] = self.rng.choice([2, 3, 4])
            elif param == "K":
                config["K"] = self.rng.choice([5, 8, 10, 12, 15])
            elif param == "alpha":
                config["alpha"] = self.rng.choice([0.05, 0.1, 0.15, 0.2])
            elif param == "theta":
                config["theta"] = self.rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        else:
            if "embedding_dim" in config:
                config["embedding_dim"] = self.rng.choice([32, 64, 128])
            if "lr" in config:
                config["lr"] = self.rng.choice([0.0005, 0.001, 0.005, 0.01])

        return config

    def should_stop(self, memory: ExperimentMemory, max_no_improve: int = 5) -> bool:
        """判断是否应该停止实验。"""
        if len(memory.records) < 3:
            return False

        metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"
        recent = memory.recent(max_no_improve + 1)
        if len(recent) < max_no_improve + 1:
            return False

        no_improve_count = 0
        current_best = recent[0].metrics.get(metric_key, 0)
        for r in recent[1:]:
            if r.metrics.get(metric_key, 0) > current_best + 1e-6:
                current_best = r.metrics.get(metric_key, 0)
                no_improve_count = 0
            else:
                no_improve_count += 1

        return no_improve_count >= max_no_improve

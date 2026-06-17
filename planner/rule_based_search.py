"""简化搜索规划器：前100轮仅使用 Optuna，无提升后再启用 Refection。

解决 Bandit + Reflection + Planner 过重的问题。
"""
import random

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class RuleBasedSearch:
    """简化搜索规划器。

    前 100 轮仅使用 Optuna 进行搜索。
    连续 10 轮无提升后，调用 Refection 辅助判断。
    """

    def __init__(
        self,
        task_type: str = "classification",
        seed: int = 42,
    ):
        self.task_type = task_type
        self.rng = random.Random(seed)
        self.round_count = 0
        self.no_improve_count = 0
        self.best_metric = 0.0
        self.use_reflection = False

        # Optuna study
        if HAS_OPTUNA:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
        else:
            self.study = None

    def next_config(self, memory: list) -> dict:
        """生成下一轮配置。

        前100轮使用 Optuna 采样，之后根据情况使用 Refection。
        """
        self.round_count += 1
        planner = "rule"
        reason = "early_stage"

        if self.round_count >= 100 and self.use_reflection:
            planner = "reflection"
            reason = "no_improve_10_rounds"

        if self.task_type == "classification":
            config = self._sample_cls_config()
        else:
            config = self._sample_rec_config()

        config["_planner"] = planner
        config["_planner_reason"] = reason

        print(f"[SEARCH] planner={planner} reason={reason} round={self.round_count}")
        return config

    def update(self, metric: float):
        """更新状态，追踪无提升轮次。"""
        if metric > self.best_metric + 1e-6:
            self.best_metric = metric
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= 10:
            self.use_reflection = True

    def _sample_cls_config(self) -> dict:
        """分类任务配置采样。"""
        if self.study is not None:
            trial = self.study.ask()
            return {
                "model_type": "Hybrid",
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 400, 500, 600]),
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 6, 7, 8, 10]),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 30, 50, 100]),
            }
        # fallback: 随机采样
        return {
            "model_type": "Hybrid",
            "n_estimators": self.rng.choice([100, 200, 300, 400, 500]),
            "max_depth": self.rng.choice([3, 5, 6, 7, 8]),
            "learning_rate": self.rng.choice([0.005, 0.01, 0.02, 0.03, 0.05]),
            "subsample": self.rng.choice([0.6, 0.7, 0.8, 0.9]),
            "colsample_bytree": self.rng.choice([0.6, 0.7, 0.8, 0.9]),
            "reg_alpha": self.rng.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
            "reg_lambda": self.rng.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
            "min_child_samples": self.rng.choice([5, 10, 20, 30, 50]),
        }

    def _sample_rec_config(self) -> dict:
        """推荐任务配置采样。"""
        return {"model_type": self.rng.choice(["Popularity", "ItemCF", "BPR_MF", "SASRec", "LightGCN"])}

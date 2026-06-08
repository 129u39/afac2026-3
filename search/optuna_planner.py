"""Optuna Planner：使用 Optuna 进行超参搜索。"""

import os
import json

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class OptunaPlanner:
    """基于 Optuna 的超参搜索规划器。

    与 BanditPlanner 配合使用：
    - BanditPlanner 负责选择模型架构
    - OptunaPlanner 负责在选定模型内搜索最优超参
    """

    def __init__(
        self,
        task_type: str,
        study_name: str | None = None,
        storage: str | None = None,
        output_dir: str = "output",
    ):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            study_name: Optuna Study 名称
            storage: Optuna 存储路径（None 为内存）
            output_dir: Study 保存目录
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna is required. Install with: pip install optuna")

        self.task_type = task_type
        self.output_dir = output_dir
        self.study_name = study_name or f"afac_{task_type}"

        direction = "maximize"  # accuracy 和 ndcg 都是越大越好
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self._trial_counter = 0
        self._current_model_type: str | None = None

    def set_model_type(self, model_type: str):
        """设置当前搜索的模型类型（由 Bandit 传入）。"""
        self._current_model_type = model_type

    def next_config(self) -> dict:
        """从 Study 中采样下一组配置。

        返回:
            config dict
        """
        trial = self.study.ask()

        # 设置模型类型（如果由 Bandit 指定）
        if self._current_model_type:
            trial.set_user_attr("model_type", self._current_model_type)

        config = self._sample_config(trial)
        config["_optuna_trial_number"] = trial.number
        return config

    def update_result(self, trial_number: int, value: float):
        """更新 trial 的结果。

        Args:
            trial_number: trial 编号
            value: 目标值（accuracy 或 ndcg）
        """
        trial = self.study.trials[trial_number]
        self.study.tell(trial, value)
        self._trial_counter += 1

    def update_result_by_config(self, config: dict, value: float):
        """通过配置中的 trial_number 更新结果。

        Args:
            config: 包含 _optuna_trial_number 的配置
            value: 目标值
        """
        trial_number = config.get("_optuna_trial_number")
        if trial_number is not None:
            self.update_result(trial_number, value)

    def best_config(self) -> dict | None:
        """返回最佳配置。"""
        if not self.study.trials:
            return None
        best_trial = self.study.best_trial
        return best_trial.params

    def best_value(self) -> float:
        """返回最佳目标值。"""
        if not self.study.trials:
            return 0.0
        return self.study.best_value

    def save(self, filename: str | None = None):
        """保存 Study 到 JSON 文件。"""
        if filename is None:
            filename = f"optuna_study_{self.task_type}.json"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        # 序列化 trials 信息
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "user_attrs": dict(trial.user_attrs),
            })

        data = {
            "study_name": self.study_name,
            "direction": "maximize",
            "best_value": self.study.best_value if self.study.trials else None,
            "best_params": self.study.best_params if self.study.trials else None,
            "trials": trials_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _sample_config(self, trial) -> dict:
        """从 trial 中采样配置。"""
        if self.task_type == "classification":
            return self._sample_cls_config(trial)
        else:
            return self._sample_rec_config(trial)

    def _sample_cls_config(self, trial) -> dict:
        """分类任务采样。"""
        return {
            "model_type": trial.suggest_categorical("model_type", ["GCN", "GAT", "GraphSAGE"]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 2, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "epochs": 200,
            "patience": 30,
        }

    def _sample_rec_config(self, trial) -> dict:
        """推荐任务采样。"""
        model_type = self._current_model_type or "BPR_MF"

        if model_type in ("Popularity", "ItemCF"):
            return {"model_type": model_type}

        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        config = {
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": 50,
        }

        if model_type in ("BPR_MF", "LightGCN"):
            config["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
        elif model_type == "SASRec":
            config["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])

        if model_type == "LightGCN":
            config["num_layers"] = trial.suggest_int("num_layers", 2, 4)

        return config

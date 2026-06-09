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

    对推荐任务，使用固定搜索空间避免动态空间问题。
    """

    def __init__(
        self,
        task_type: str,
        study_name: str | None = None,
        storage: str | None = None,
        output_dir: str = "output",
    ):
        if not HAS_OPTUNA:
            raise ImportError("optuna is required. Install with: pip install optuna")

        self.task_type = task_type
        self.output_dir = output_dir
        self.study_name = study_name or f"afac_{task_type}"

        direction = "maximize"
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
        """从 Study 中采样下一组配置。"""
        trial = self.study.ask()

        if self._current_model_type:
            trial.set_user_attr("model_type", self._current_model_type)

        config = self._sample_config(trial)
        config["_optuna_trial_number"] = trial.number
        return config

    def update_result(self, trial_number: int, value: float):
        """更新 trial 的结果。"""
        try:
            self.study.tell(trial_number, value)
            self._trial_counter += 1
        except (ValueError, RuntimeError):
            pass  # trial 已完成或不存在，跳过

    def update_result_by_config(self, config: dict, value: float):
        """通过配置中的 trial_number 更新结果。"""
        trial_number = config.get("_optuna_trial_number")
        if trial_number is not None:
            self.update_result(trial_number, value)

    def best_config(self) -> dict | None:
        """返回最佳配置。"""
        if not self.study.trials:
            return None
        return self.study.best_trial.params

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
        """分类任务采样。

        V4: 支持 GraphSAGE, APPNP, GCNII, MLP 等多种模型。
        """
        import random

        # V4: 模型选择概率
        model_weights = {
            "GraphSAGE": 0.5,
            "APPNP": 0.2,
            "GCNII": 0.15,
            "GCN": 0.05,
            "GAT": 0.05,
            "MLP": 0.05,
        }
        model_type = random.choices(
            list(model_weights.keys()),
            weights=list(model_weights.values()),
            k=1
        )[0]

        # 根据模型类型选择搜索空间
        if model_type == "GraphSAGE":
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            num_layers = trial.suggest_int("num_layers", 2, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
            lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        elif model_type == "APPNP":
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 2, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
            lr = trial.suggest_float("lr", 1e-3, 5e-3, log=True)
            K = trial.suggest_int("K", 5, 15)
            alpha = trial.suggest_float("alpha", 0.05, 0.2, step=0.05)
        elif model_type == "GCNII":
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 4, 16)
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
            lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
            alpha = trial.suggest_float("alpha", 0.05, 0.2, step=0.05)
            theta = trial.suggest_float("theta", 0.3, 0.7, step=0.1)
        elif model_type == "MLP":
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 2, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
            lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        else:  # GCN, GAT
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 2, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
            lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)

        config = {
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "epochs": 200,
            "patience": 30,
        }

        # 添加模型特有参数
        if model_type == "APPNP":
            config["K"] = K
            config["alpha"] = alpha
        elif model_type == "GCNII":
            config["alpha"] = alpha
            config["theta"] = theta

        return config

    def _sample_rec_config(self, trial) -> dict:
        """推荐任务采样。

        使用固定搜索空间，避免模型切换时 Optuna 动态空间报错。
        对于简单模型（Popularity/ItemCF），只取 model_type，其余参数忽略。
        """
        model_type = self._current_model_type or "BPR_MF"

        if model_type in ("Popularity", "ItemCF"):
            # 简单模型无超参，只记录 trial
            return {"model_type": model_type}

        # 统一搜索空间：所有可调模型共用同一套参数名
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        # 统一 batch_size 选项（取所有模型的并集）
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        # num_layers 仅 LightGCN 使用，其他模型忽略
        num_layers = trial.suggest_int("num_layers", 2, 4)

        config = {
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": 50,
        }

        if model_type == "LightGCN":
            config["num_layers"] = num_layers

        return config

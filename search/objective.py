"""Optuna 目标函数：封装模型训练与评估。"""

from typing import Callable


def create_classification_objective(run_experiment_fn: Callable) -> Callable:
    """创建分类任务的 Optuna 目标函数。

    Args:
        run_experiment_fn: 执行实验的函数，接受 config dict，返回 {"metrics": {...}}

    返回:
        Optuna 目标函数
    """
    def objective(trial):
        config = {
            "model_type": trial.suggest_categorical("model_type", ["GCN", "GAT", "GraphSAGE"]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 2, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "epochs": 200,
            "patience": 30,
        }

        try:
            result = run_experiment_fn(config)
            return result["metrics"].get("val_accuracy", 0.0)
        except Exception as e:
            # 实验失败，返回 0
            trial.set_user_attr("error", str(e))
            return 0.0

    return objective


def create_recommendation_objective(run_experiment_fn: Callable) -> Callable:
    """创建推荐任务的 Optuna 目标函数。

    注意：模型类型通过 user_attrs 传入（由 Bandit 决定）。
    """
    def objective(trial):
        model_type = trial.user_attrs.get("model_type", "BPR_MF")

        if model_type in ("Popularity", "ItemCF"):
            config = {"model_type": model_type}
        else:
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

        try:
            result = run_experiment_fn(config)
            return result["metrics"].get("ndcg@k", 0.0)
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return 0.0

    return objective

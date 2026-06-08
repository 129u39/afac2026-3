"""Optuna 搜索空间定义。"""


def classification_search_space(trial) -> dict:
    """分类任务搜索空间。

    每个 trial 从搜索空间中采样一组超参。
    """
    model_type = trial.suggest_categorical("model_type", ["GCN", "GAT", "GraphSAGE"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    return {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": 200,
        "patience": 30,
    }


def recommendation_search_space(trial) -> dict:
    """推荐任务搜索空间。

    注意：不搜索模型类型（由 Bandit 决定），只搜索超参。
    模型类型通过 trial.user_attrs 传入。
    """
    model_type = trial.user_attrs.get("model_type", "BPR_MF")

    if model_type == "Popularity" or model_type == "ItemCF":
        # 这两个模型没有可调超参
        return {"model_type": model_type}

    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)

    config = {
        "model_type": model_type,
        "embedding_dim": embedding_dim,
        "lr": lr,
    }

    if model_type == "BPR_MF":
        config["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        config["epochs"] = 50

    elif model_type == "SASRec":
        config["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        config["epochs"] = 50

    elif model_type == "LightGCN":
        config["num_layers"] = trial.suggest_int("num_layers", 2, 4)
        config["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        config["epochs"] = 50

    return config

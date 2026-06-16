"""统一实验运行器：封装分类和推荐实验的执行。"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from models.utils import set_seed, get_device, enable_amp, enable_tf32
from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn
from models.recommender import RecommenderSystem
from data_loader import classification_to_pyg
from evaluate import evaluate_classification, evaluate_recommendation

# 新模型
try:
    from models.gcnii import GCNII, train_gcnii
    from models.mlp_baseline import MLPBaseline, train_mlp
    from models.lightgbm_model import LightGBMModel
    from models.xgboost_model import XGBoostModel
    from features.lgb_selector import LGBSelector
    from features.xgb_selector import XGBSelector
    HAS_NEW_MODELS = True
except ImportError:
    HAS_NEW_MODELS = False


@dataclass
class ExperimentResult:
    """实验结果。"""
    metric: float
    train_time: float
    model_name: str
    config: dict
    metrics: dict = field(default_factory=dict)
    model: Any = None
    status: str = "success"
    error: str = ""


class ExperimentRunner:
    """统一实验运行器。

    封装分类和推荐实验的执行逻辑，提供统一的 run(config) 接口。
    """

    def __init__(self, task_type: str, data: dict, device: torch.device | None = None):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            data: 已加载的数据字典
            device: 计算设备
        """
        self.task_type = task_type
        self.data = data
        self.device = device or get_device()

    def run(self, config: dict) -> ExperimentResult:
        """执行实验。

        Args:
            config: 实验配置，必须包含 model_type

        返回:
            ExperimentResult
        """
        model_type = config.get("model_type", "unknown")
        start_time = time.time()

        try:
            if self.task_type == "classification":
                result = self._run_classification(config)
            else:
                result = self._run_recommendation(config)

            result.train_time = time.time() - start_time
            return result

        except Exception as e:
            return ExperimentResult(
                metric=0.0,
                train_time=time.time() - start_time,
                model_name=model_type,
                config=config,
                status="failed",
                error=str(e),
            )

    def fit_full(self, config: dict):
        """用全部训练数据重训最终提交模型。"""
        if self.task_type == "classification":
            return self._fit_full_classification(config)
        return self._fit_full_recommendation(config)

    def _prepare_classification_inputs(self, config: dict):
        data = self.data
        pyg_data = classification_to_pyg(data, self.device)

        features = data["features"].toarray()
        labels = data["labels"]
        train_idx = data["train_idx"]

        feature_selector_type = config.get("feature_selector", "none")
        feature_dim = config.get("feature_dim", 256)

        if feature_selector_type == "lgb" and HAS_NEW_MODELS:
            selector = LGBSelector(n_top_features=feature_dim)
            selector.fit(features[train_idx], labels[train_idx])
            all_features = selector.transform(data["features"].toarray())
        elif feature_selector_type == "xgb" and HAS_NEW_MODELS:
            selector = XGBSelector(n_top_features=feature_dim)
            selector.fit(features[train_idx], labels[train_idx])
            all_features = selector.transform(data["features"].toarray())
        else:
            all_features = features

        return pyg_data, all_features, labels, train_idx

    def _fit_full_classification(self, config: dict):
        data = self.data
        pyg_data, all_features, labels, train_idx = self._prepare_classification_inputs(config)
        model_type = config.get("model_type", "MLP")

        if model_type == "LightGBM" and HAS_NEW_MODELS:
            model = LightGBMModel(
                n_estimators=config.get("n_estimators", 500),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", config.get("lr", 0.05)),
            )
            model.fit(all_features[train_idx], labels[train_idx])
            return model

        if model_type == "GCNII" and HAS_NEW_MODELS:
            model = GCNII(
                in_dim=all_features.shape[1],
                hidden_dim=config.get("hidden_dim", 256),
                num_classes=data["num_classes"],
                num_layers=config.get("num_layers", 16),
                dropout=config.get("dropout", 0.3),
                alpha=config.get("alpha", 0.1),
                theta=config.get("theta", 0.5),
            ).to(self.device)
            train_gcnii(
                model, pyg_data,
                lr=config.get("lr", 0.01),
                weight_decay=config.get("weight_decay", 5e-4),
                epochs=max(20, config.get("epochs", 200) // 2),
                patience=config.get("patience", 30),
                full_train=True,
            )
            return model

        if model_type == "MLP" and HAS_NEW_MODELS:
            model = MLPBaseline(
                in_dim=all_features.shape[1],
                hidden_dim=config.get("hidden_dim", 512),
                num_classes=data["num_classes"],
                num_layers=config.get("num_layers", 2),
                dropout=config.get("dropout", 0.3),
            ).to(self.device)
            train_mlp(
                model, pyg_data,
                lr=config.get("lr", 0.01),
                weight_decay=config.get("weight_decay", 5e-4),
                epochs=max(20, config.get("epochs", 200) // 2),
                patience=config.get("patience", 30),
                full_train=True,
            )
            return model

        model = GNNClassifier(
            in_dim=data["num_features"],
            hidden_dim=config.get("hidden_dim", 128),
            num_classes=data["num_classes"],
            num_layers=config.get("num_layers", 3),
            model_type=model_type,
            dropout=config.get("dropout", 0.1),
        ).to(self.device)
        train_gnn(
            model, pyg_data,
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4),
            epochs=max(20, config.get("epochs", 200) // 2),
            patience=config.get("patience", 30),
            full_train=True,
        )
        return model

    def _fit_full_recommendation(self, config: dict):
        model_type = config.get("model_type", "Popularity")
        kwargs = {k: v for k, v in config.items() if k != "model_type"}
        rec_sys = RecommenderSystem(model_type=model_type, **kwargs)
        rec_sys.fit(self.data)
        return rec_sys

    def _run_classification(self, config: dict) -> ExperimentResult:
        """执行分类实验。"""
        import numpy as np
        data = self.data
        pyg_data = classification_to_pyg(data, self.device)
        model_type = config.get("model_type", "MLP")

        # 启用 AMP 和 TF32
        enable_tf32()

        # 特征筛选
        features = data["features"].toarray()
        labels = data["labels"]
        train_idx = data["train_idx"]

        feature_selector_type = config.get("feature_selector", "none")
        feature_dim = config.get("feature_dim", 256)

        if feature_selector_type == "lgb" and HAS_NEW_MODELS:
            selector = LGBSelector(n_top_features=feature_dim)
            features = selector.fit_transform(features[train_idx], labels[train_idx])
            # 对全量数据转换
            all_features = selector.transform(data["features"].toarray())
        elif feature_selector_type == "xgb" and HAS_NEW_MODELS:
            selector = XGBSelector(n_top_features=feature_dim)
            features = selector.fit_transform(features[train_idx], labels[train_idx])
            all_features = selector.transform(data["features"].toarray())
        else:
            all_features = features

        # 根据模型类型创建模型
        if model_type == "LightGBM" and HAS_NEW_MODELS:
            # LightGBM 直接训练
            model = LightGBMModel(
                n_estimators=config.get("n_estimators", 500),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", config.get("lr", 0.05)),
            )
            train_features = all_features[train_idx]
            train_labels = labels[train_idx]

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )

            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            from sklearn.metrics import accuracy_score
            val_acc = accuracy_score(y_val, val_pred)

            return ExperimentResult(
                metric=val_acc,
                train_time=0.0,
                model_name=model_type,
                config=config,
                metrics={"val_accuracy": val_acc},
                model=model,
            )
        elif model_type == "GCNII" and HAS_NEW_MODELS:
            model = GCNII(
                in_dim=all_features.shape[1],
                hidden_dim=config.get("hidden_dim", 256),
                num_classes=data["num_classes"],
                num_layers=config.get("num_layers", 16),
                dropout=config.get("dropout", 0.3),
                alpha=config.get("alpha", 0.1),
                theta=config.get("theta", 0.5),
            ).to(self.device)
            train_result = train_gcnii(
                model, pyg_data,
                lr=config.get("lr", 0.01),
                weight_decay=config.get("weight_decay", 5e-4),
                epochs=config.get("epochs", 200),
                patience=config.get("patience", 30),
            )
        elif model_type == "MLP" and HAS_NEW_MODELS:
            model = MLPBaseline(
                in_dim=all_features.shape[1],
                hidden_dim=config.get("hidden_dim", 512),
                num_classes=data["num_classes"],
                num_layers=config.get("num_layers", 2),
                dropout=config.get("dropout", 0.3),
            ).to(self.device)
            train_result = train_mlp(
                model, pyg_data,
                lr=config.get("lr", 0.01),
                weight_decay=config.get("weight_decay", 5e-4),
                epochs=config.get("epochs", 200),
                patience=config.get("patience", 30),
            )
        else:
            # GCN, GAT, GraphSAGE
            model = GNNClassifier(
                in_dim=data["num_features"],
                hidden_dim=config.get("hidden_dim", 128),
                num_classes=data["num_classes"],
                num_layers=config.get("num_layers", 3),
                model_type=model_type,
                dropout=config.get("dropout", 0.1),
            ).to(self.device)
            train_result = train_gnn(
                model, pyg_data,
                lr=config.get("lr", 0.005),
                weight_decay=config.get("weight_decay", 5e-4),
                epochs=config.get("epochs", 200),
                patience=config.get("patience", 30),
            )

        eval_result = evaluate_classification(
            model, pyg_data, data["train_idx"], data["labels"]
        )

        return ExperimentResult(
            metric=eval_result["val_accuracy"],
            train_time=0.0,  # 由调用方设置
            model_name=model_type,
            config=config,
            metrics={
                "val_accuracy": eval_result["val_accuracy"],
                "train_loss": train_result["train_losses"][-1] if train_result["train_losses"] else 0,
            },
            model=model,
        )

    def _run_recommendation(self, config: dict) -> ExperimentResult:
        """执行推荐实验。"""
        model_type = config.get("model_type", "Popularity")
        kwargs = {k: v for k, v in config.items() if k != "model_type"}

        rec_sys = RecommenderSystem(model_type=model_type, **kwargs)
        rec_sys.fit(self.data)

        eval_result = evaluate_recommendation(rec_sys, self.data)

        return ExperimentResult(
            metric=eval_result["ndcg@k"],
            train_time=0.0,  # 由调用方设置
            model_name=model_type,
            config=config,
            metrics={
                "ndcg@k": eval_result["ndcg@k"],
                "hit@k": eval_result["hit@k"],
            },
            model=rec_sys,
        )

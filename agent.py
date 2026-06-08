"""Agent 主控：完整的实验循环。"""

import time
import numpy as np
import torch

from models.utils import set_seed, get_device
from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn
from models.recommender import RecommenderSystem
from data_loader import load_classification, classification_to_pyg, load_recommendation, parse_seq_dedup
from evaluate import evaluate_classification, evaluate_recommendation
from memory import ExperimentMemory, ExperimentRecord
from planner import Planner
from budget_manager import BudgetManager
from feedback_analyzer import FeedbackAnalyzer
from trajectory_logger import TrajectoryLogger
import config


class Agent:
    """自动化实验 Agent。"""

    def __init__(self, task_type: str, data_path: str, output_dir: str = config.OUTPUT_DIR):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            data_path: 数据路径（分类为 .npz 路径，推荐为目录路径）
            output_dir: 输出目录
        """
        self.task_type = task_type
        self.data_path = data_path
        self.output_dir = output_dir

        set_seed(config.SEED)
        self.device = get_device()
        self.memory = ExperimentMemory()
        self.planner = Planner(task_type, seed=config.SEED)
        self.budget = BudgetManager(config.TOTAL_BUDGET_SECONDS, config.SAFETY_MARGIN_SECONDS)
        self.analyzer = FeedbackAnalyzer()
        self.logger = TrajectoryLogger(output_dir, task_name=task_type)

        self._best_model = None
        self._best_metric = 0.0
        self._data = None

    def load_data(self):
        """加载数据。"""
        print(f"[Agent] 加载 {self.task_type} 数据...")
        if self.task_type == "classification":
            self._data = load_classification(self.data_path)
            print(f"  节点数: {self._data['num_nodes']}, "
                  f"特征维度: {self._data['num_features']}, "
                  f"类别数: {self._data['num_classes']}, "
                  f"训练节点: {len(self._data['train_idx'])}, "
                  f"测试节点: {len(self._data['test_idx'])}")
        else:
            self._data = load_recommendation(self.data_path)
            print(f"  用户数: {len(self._data['train_df']) + len(self._data['test_df'])}, "
                  f"物品数: {self._data['num_items']}, "
                  f"训练集: {len(self._data['train_df'])}, "
                  f"测试集: {len(self._data['test_df'])}")

    def run(self) -> dict:
        """运行完整的实验循环。

        返回:
            {"best_metric": float, "best_config": dict, "num_rounds": int}
        """
        self.load_data()
        print(f"\n[Agent] 开始实验循环，预算: {self.budget.total_seconds}s")
        print(self.budget.format_status())

        round_num = 0
        while self.budget.has_time(self.budget.avg_round_time()):
            # 1. 规划下一轮
            config_dict = self.planner.next_config(self.memory)
            print(f"\n{'='*60}")
            print(f"Round {round_num}: {config_dict.get('model_type', 'unknown')}")
            print(f"  Config: {config_dict}")

            # 2. 执行实验
            round_start = time.time()
            try:
                result = self._run_experiment(config_dict)
            except Exception as e:
                print(f"  [ERROR] 实验失败: {e}")
                round_num += 1
                continue
            round_time = time.time() - round_start
            self.budget.record_round(round_time)

            # 3. 记录
            record = ExperimentRecord(
                round_num=round_num,
                model_type=config_dict.get("model_type", "unknown"),
                config=config_dict,
                metrics=result["metrics"],
                elapsed_seconds=round_time,
            )
            self.memory.record(record)

            # 4. 更新最佳模型
            metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"
            current_metric = result["metrics"].get(metric_key, 0)
            if current_metric > self._best_metric:
                self._best_metric = current_metric
                self._best_model = result.get("model")
                print(f"  ★ 新最佳! {metric_key}={current_metric:.4f}")

            # 5. 分析反馈
            feedback = self.analyzer.analyze(self.memory, self.task_type)
            strategy = " | ".join(feedback["suggestions"][:2]) if feedback["suggestions"] else ""

            # 6. 记录日志
            self.logger.log(
                round_num=round_num,
                config=config_dict,
                metrics=result["metrics"],
                feedback={"trend": feedback["trend"], "suggestions": feedback["suggestions"]},
                strategy=strategy,
                elapsed_seconds=round_time,
            )

            print(f"  耗时: {round_time:.1f}s | {metric_key}: {current_metric:.4f}")
            print(f"  趋势: {feedback['trend']} | 建议: {strategy}")
            print(self.budget.format_status())

            # 7. 检查是否应该停止
            if self.planner.should_stop(self.memory):
                print("\n[Agent] 连续多轮无提升，停止实验")
                break

            round_num += 1

        # 保存日志
        self.logger.save()
        print(f"\n[Agent] 实验结束，共 {round_num} 轮")
        print(self.memory.summary())

        return {
            "best_metric": self._best_metric,
            "best_config": self.memory.best_classification().config if self.task_type == "classification" and self.memory.best_classification() else (self.memory.best_recommendation().config if self.memory.best_recommendation() else {}),
            "num_rounds": round_num,
        }

    def _run_experiment(self, config_dict: dict) -> dict:
        """执行单轮实验。"""
        if self.task_type == "classification":
            return self._run_cls_experiment(config_dict)
        else:
            return self._run_rec_experiment(config_dict)

    def _run_cls_experiment(self, config_dict: dict) -> dict:
        """执行分类实验。"""
        data = self._data
        pyg_data = classification_to_pyg(data, self.device)

        model = GNNClassifier(
            in_dim=data["num_features"],
            hidden_dim=config_dict.get("hidden_dim", 64),
            num_classes=data["num_classes"],
            num_layers=config_dict.get("num_layers", 2),
            model_type=config_dict.get("model_type", "GCN"),
            dropout=config_dict.get("dropout", 0.5),
        ).to(self.device)

        result = train_gnn(
            model, pyg_data,
            lr=config_dict.get("lr", 0.01),
            weight_decay=config_dict.get("weight_decay", 5e-4),
            epochs=config_dict.get("epochs", 200),
            patience=config_dict.get("patience", 30),
        )

        # 本地验证
        eval_result = evaluate_classification(
            model, pyg_data, data["train_idx"], data["labels"]
        )

        return {
            "model": model,
            "metrics": {
                "val_accuracy": eval_result["val_accuracy"],
                "train_loss": result["train_losses"][-1] if result["train_losses"] else 0,
            },
        }

    def _run_rec_experiment(self, config_dict: dict) -> dict:
        """执行推荐实验。"""
        model_type = config_dict.get("model_type", "Popularity")
        kwargs = {k: v for k, v in config_dict.items() if k != "model_type"}

        rec_sys = RecommenderSystem(model_type=model_type, **kwargs)
        rec_sys.fit(self._data)

        # 本地验证
        eval_result = evaluate_recommendation(rec_sys, self._data)

        return {
            "model": rec_sys,
            "metrics": {
                "ndcg@k": eval_result["ndcg@k"],
                "hit@k": eval_result["hit@k"],
            },
        }

    def generate_submission(self, output_path: str):
        """基于最佳模型生成提交文件。"""
        if self._best_model is None:
            print("[Agent] 没有可用的最佳模型，跳过提交生成")
            return

        print(f"\n[Agent] 生成提交文件: {output_path}")

        if self.task_type == "classification":
            self._generate_cls_submission(output_path)
        else:
            self._generate_rec_submission(output_path)

    def _generate_cls_submission(self, output_path: str):
        """生成分类提交文件。"""
        data = self._data
        pyg_data = classification_to_pyg(data, self.device)

        predictions = predict_gnn(self._best_model, pyg_data)
        test_idx = data["test_idx"]

        import pandas as pd
        df = pd.DataFrame({"test_idx": test_idx, "label": predictions.numpy()})
        df.to_csv(output_path, index=False)
        print(f"  提交文件已生成: {output_path} ({len(df)} 行)")

    def _generate_rec_submission(self, output_path: str):
        """生成推荐提交文件。"""
        test_df = self._data["test_df"]
        results = []

        for _, row in test_df.iterrows():
            uid = row["uid"]
            seq = parse_seq_dedup(row["item_seq_dedup"])
            pred_list = self._best_model.predict(uid, seq, top_k=10)
            results.append({"uid": uid, "prediction": ",".join(pred_list)})

        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"  提交文件已生成: {output_path} ({len(df)} 行)")

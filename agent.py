"""Agent 主控 V1：Bandit 选模型 + Optuna 调超参 + Reflection 评估。"""

import time
import numpy as np
import torch

from models.utils import set_seed, get_device
from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn
from models.recommender import RecommenderSystem
from data_loader import load_classification, classification_to_pyg, load_recommendation, parse_seq_dedup
from evaluate import evaluate_classification, evaluate_recommendation
from memory import ExperimentMemory, ExperimentRecord
from memory.retriever import Retriever
from planner.bandit_planner import BanditPlanner
from budget_manager import BudgetManager
from feedback_analyzer import FeedbackAnalyzer
from analysis.reflection import ReflectionAgent
from trajectory_logger import TrajectoryLogger
import config

# Optuna 可选导入
try:
    from search.optuna_planner import OptunaPlanner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class Agent:
    """自动化实验 Agent V1。

    架构:
    1. BanditPlanner — 选择模型架构（UCB exploration/exploitation）
    2. OptunaPlanner — 搜索最优超参（在选定模型内）
    3. Retriever — 检索相似历史实验作为参考
    4. ReflectionAgent — 反思实验结果，指导下一步决策
    5. BudgetManager — 预算感知 + 早停
    """

    def __init__(
        self,
        task_type: str,
        data_path: str,
        output_dir: str = config.OUTPUT_DIR,
        use_optuna: bool = True,
    ):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            data_path: 数据路径
            output_dir: 输出目录
            use_optuna: 是否使用 Optuna 进行超参搜索
        """
        self.task_type = task_type
        self.data_path = data_path
        self.output_dir = output_dir

        set_seed(config.SEED)
        self.device = get_device()

        # 核心组件
        self.memory = ExperimentMemory()
        self.budget = BudgetManager(
            config.TOTAL_BUDGET_SECONDS,
            config.SAFETY_MARGIN_SECONDS,
        )
        self.analyzer = FeedbackAnalyzer()
        self.logger = TrajectoryLogger(output_dir, task_name=task_type)

        # V1 新组件
        self.bandit_planner = BanditPlanner(task_type, seed=config.SEED)
        self.reflection = ReflectionAgent()
        self.retriever = Retriever(task_type)

        # Optuna 可选
        self.optuna_planner = None
        if use_optuna and HAS_OPTUNA:
            try:
                self.optuna_planner = OptunaPlanner(
                    task_type=task_type,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"[Agent] Optuna 初始化失败，回退到 Bandit: {e}")

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
        print(f"\n[Agent] 开始 V1 实验循环，预算: {self.budget.total_seconds}s")
        print(f"  Bandit: UCB (c=1.5)")
        print(f"  Optuna: {'启用' if self.optuna_planner else '未启用'}")
        print(f"  Retriever: 启用")
        print(f"  Reflection: 启用")
        print(self.budget.format_status())

        round_num = 0
        while self.budget.should_continue():
            # 1. 规划下一轮配置
            config_dict = self._plan_next_config()
            model_type = config_dict.get("model_type", "unknown")
            print(f"\n{'='*60}")
            print(f"Round {round_num}: {model_type}")
            print(f"  Config: {config_dict}")

            # 2. 检查是否与已有实验过于相似
            if self.retriever.is_too_similar(config_dict, threshold=0.99):
                print("  [SKIP] 配置与已有实验过于相似，跳过")
                # 对 Optuna 报告一个低分
                if self.optuna_planner and "_optuna_trial_number" in config_dict:
                    self.optuna_planner.update_result(
                        config_dict["_optuna_trial_number"], 0.0
                    )
                round_num += 1
                continue

            # 3. 执行实验
            round_start = time.time()
            try:
                result = self._run_experiment(config_dict)
            except Exception as e:
                print(f"  [ERROR] 实验失败: {e}")
                # 对 Optuna 报告失败
                if self.optuna_planner and "_optuna_trial_number" in config_dict:
                    self.optuna_planner.update_result(
                        config_dict["_optuna_trial_number"], 0.0
                    )
                # 记录失败实验
                record = ExperimentRecord(
                    round_num=round_num,
                    model_type=model_type,
                    config=config_dict,
                    metrics={},
                    elapsed_seconds=time.time() - round_start,
                    task=self.task_type,
                    status="failed",
                    notes=str(e),
                )
                self.memory.add(record)
                round_num += 1
                continue

            round_time = time.time() - round_start
            self.budget.record_round(round_time)

            # 4. 记录实验
            metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"
            current_metric = result["metrics"].get(metric_key, 0)
            improved = current_metric > self._best_metric

            record = ExperimentRecord(
                round_num=round_num,
                model_type=model_type,
                config=config_dict,
                metrics=result["metrics"],
                elapsed_seconds=round_time,
                task=self.task_type,
                status="success",
            )
            self.memory.add(record)

            # 更新预算管理器的改进追踪
            self.budget.record_improvement(improved)

            # 5. 更新最佳模型
            if improved:
                self._best_metric = current_metric
                self._best_model = result.get("model")
                print(f"  ★ 新最佳! {metric_key}={current_metric:.4f}")

            # 6. 更新 Optuna
            if self.optuna_planner and "_optuna_trial_number" in config_dict:
                self.optuna_planner.update_result(
                    config_dict["_optuna_trial_number"], current_metric
                )

            # 7. 更新 Retriever 索引
            self.retriever.index(self.memory)

            # 8. 分析反馈
            feedback = self.analyzer.analyze(self.memory, self.task_type)

            # 9. 反思
            reflection = self.reflection.reflect(self.memory, feedback, self.task_type)
            print(f"  反思: {reflection['observation'][:80]}...")
            print(f"  建议: {reflection['next_action']} (置信度: {reflection['confidence']:.2f})")

            # 10. 记录日志
            decision = {
                "bandit_arm": model_type,
                "bandit_stats": self.bandit_planner.bandit.get_stats(),
                "reflection": reflection,
            }
            strategy = " | ".join(feedback["suggestions"][:2]) if feedback["suggestions"] else ""
            self.logger.log(
                round_num=round_num,
                config=config_dict,
                metrics=result["metrics"],
                feedback={"trend": feedback["trend"], "suggestions": feedback["suggestions"]},
                strategy=strategy,
                elapsed_seconds=round_time,
                decision=decision,
                runtime={"device": str(self.device), "elapsed_seconds": round_time},
            )

            print(f"  耗时: {round_time:.1f}s | {metric_key}: {current_metric:.4f}")
            print(f"  趋势: {feedback['trend']} | 风险: {feedback.get('risk', 'none')}")
            print(f"  建议: {strategy}")
            print(self.budget.format_status())

            round_num += 1

        # 保存日志和 Optuna Study
        self.logger.save()
        if self.optuna_planner:
            self.optuna_planner.save()

        # 保存实验记忆
        memory_path = f"{self.output_dir}/memory_{self.task_type}.json"
        self.memory.save(memory_path)

        print(f"\n[Agent] 实验结束，共 {round_num} 轮")
        print(self.memory.summary())

        return {
            "best_metric": self._best_metric,
            "best_config": self.memory.get_best(self.task_type).config if self.memory.get_best(self.task_type) else {},
            "num_rounds": round_num,
        }

    def _plan_next_config(self) -> dict:
        """规划下一轮配置。

        策略:
        1. 用 Bandit 选择模型架构
        2. 如果有 Optuna，在选定模型内搜索超参
        3. 否则用 BanditPlanner 的默认/微调配置
        """
        # 先更新 Bandit 统计（从上次中断处继续）
        self.bandit_planner._update_bandit_from_memory(self.memory)

        # Bandit 选择模型
        selected_arm = self.bandit_planner.bandit.select_arm()

        # 如果有 Optuna，用 Optuna 搜索超参
        if self.optuna_planner:
            self.optuna_planner.set_model_type(selected_arm)
            config_dict = self.optuna_planner.next_config()
            config_dict["model_type"] = selected_arm
            return config_dict

        # 否则用 BanditPlanner 生成配置
        return self.bandit_planner.next_config(self.memory)

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

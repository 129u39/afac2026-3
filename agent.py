"""Agent 主控：Compute-Aware AutoML Agent。"""

import time
import torch

from models.utils import set_seed, get_device
from models.gnn_classifier import predict_gnn
from models.ensemble import EnsembleBuilder
from data_loader import load_classification, load_recommendation, parse_seq_dedup
from memory import ExperimentMemory, ExperimentRecord
from memory.retriever import Retriever
from memory.knowledge_base import KnowledgeBase, KnowledgeEntry
from planner.bandit_planner import BanditPlanner
from planner.qwen_planner import QwenPlanner
from runner.experiment_runner import ExperimentRunner
from budget_manager import BudgetManager
from feedback_analyzer import FeedbackAnalyzer
from analysis.reflection import ReflectionAgent
from analysis.pattern_extractor import PatternExtractor
from analysis.adaptive_strategy import AdaptiveStrategy
from trajectory_logger import TrajectoryLogger
import config

# 可选导入
try:
    from llm.client import QwenClient
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

try:
    from search.optuna_planner import OptunaPlanner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from search.topk_pool import TopKPool
    HAS_TOPK = True
except ImportError:
    HAS_TOPK = False


class Agent:
    """自动化实验 Agent — Compute-Aware AutoML Agent。

    架构：
    1. TopK Pool — 维护最优配置池
    2. 分层模型策略 — APPNP/GraphSAGE/GCN 分层搜索
    3. Compute-Aware Reward — reward = accuracy_gain / runtime
    4. Ensemble Builder — Top-K 模型集成

    反馈循环：
    1. FeedbackAnalyzer — 趋势分析
    2. ReflectionAgent(Qwen) — 深度反思
    3. KnowledgeBase — 跨任务经验
    """

    def __init__(
        self,
        task_type: str,
        data_path: str,
        output_dir: str = config.OUTPUT_DIR,
        use_optuna: bool = True,
        use_qwen: bool = True,
    ):
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

        # 核心组件
        self.bandit_planner = BanditPlanner(task_type, seed=config.SEED)
        self.retriever = Retriever(task_type)
        self.knowledge_base = KnowledgeBase(f"{output_dir}/knowledge_base.json")
        self.topk_pool = TopKPool(max_size=20, path=f"{output_dir}/top_pool_{task_type}.json") if HAS_TOPK else None
        self.ensemble = EnsembleBuilder(task_type)
        self.pattern_extractor = PatternExtractor()
        self.adaptive_strategy = AdaptiveStrategy()
        self._data_profile = None

        # Qwen 客户端
        self.qwen_client = None
        if use_qwen and HAS_QWEN:
            try:
                self.qwen_client = QwenClient()
                if not self.qwen_client.available:
                    print("[Agent] Qwen API Key 未设置，使用规则模式")
                    self.qwen_client = None
            except Exception as e:
                print(f"[Agent] Qwen 初始化失败: {e}")

        # 反思和规划
        self.reflection = ReflectionAgent(qwen_client=self.qwen_client)
        self.qwen_planner = QwenPlanner(qwen_client=self.qwen_client)

        # Optuna
        self.optuna_planner = None
        if use_optuna and HAS_OPTUNA:
            try:
                self.optuna_planner = OptunaPlanner(
                    task_type=task_type,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"[Agent] Optuna 初始化失败: {e}")

        # 实验运行器（数据加载后初始化）
        self.runner: ExperimentRunner | None = None
        self._best_model = None
        self._best_metric = 0.0
        self._data = None

    def load_data(self):
        """加载数据并分析特征。"""
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

        # 分析数据特征
        self._data_profile = self.adaptive_strategy.analyze(self._data, self.task_type)
        strategy = self.adaptive_strategy.get_search_strategy(self._data_profile)
        print(f"\n[Agent] 数据画像:")
        print(f"  特征稀疏度: {self._data_profile.feature_sparsity:.1%}")
        if self.task_type == "classification":
            print(f"  类别不平衡度: {self._data_profile.class_imbalance:.1f}")
            print(f"  平均度: {self._data_profile.avg_degree:.1f}")
        print(f"  推荐策略: {strategy['reasoning']}")

        self.runner = ExperimentRunner(self.task_type, self._data, self.device)

    def run(self) -> dict:
        """运行完整的实验循环。"""
        self.load_data()
        qwen_status = "启用" if self.qwen_client and self.qwen_client.available else "未启用"
        optuna_status = "启用" if self.optuna_planner else "未启用"
        topk_status = "启用" if self.topk_pool else "未启用"
        print(f"\n[Agent] 开始实验循环，预算: {self.budget.total_seconds}s")
        print(f"  Bandit: UCB (c=1.5) + Compute-Aware Reward")
        print(f"  Optuna: {optuna_status}")
        print(f"  TopK Pool: {topk_status}")
        print(f"  Qwen: {qwen_status}")
        print(self.budget.format_status())

        round_num = 0
        while self.budget.should_continue():
            # 1. Bandit 选择模型（Compute-Aware Reward）
            self.bandit_planner._update_bandit_from_memory(self.memory)
            selected_arm = self.bandit_planner.bandit.select_arm()

            # 2. 生成候选配置（优先从 TopK Pool 生成）
            candidate_configs = self._generate_candidates(selected_arm)

            # 3. 获取反思
            feedback = self.analyzer.analyze(self.memory, self.task_type)
            similar = self.retriever.top_k_similar(
                candidate_configs[0] if candidate_configs else {}, k=3
            )
            reflection_result = self.reflection.reflect(
                self.memory, feedback, self.task_type,
                similar_experiments=similar,
                budget_state=self.budget.remaining_budget(),
            )

            # 4. Qwen Planner 最终决策
            best = self.memory.get_best(self.task_type)
            best_info = {
                "model": best.model_type,
                "metric": best.metrics,
                "config": best.config,
            } if best else {}

            decision = self.qwen_planner.select(
                candidate_configs=candidate_configs,
                reflection=reflection_result.model_dump() if reflection_result else {},
                budget_state=self.budget.remaining_budget(),
                history_summary=self.memory.summary(),
                best_experiment=best_info,
                task_type=self.task_type,
            )

            config_dict = decision.selected_config
            if not config_dict:
                config_dict = candidate_configs[0] if candidate_configs else {}

            model_type = config_dict.get("model_type", selected_arm)
            print(f"\n{'='*60}")
            print(f"Round {round_num}: {model_type}")
            print(f"  Config: {config_dict}")
            print(f"  Decision: {decision.reason} (confidence={decision.confidence:.2f})")

            # 5. 检查是否过于相似
            has_hyperparams = model_type not in ("Popularity", "ItemCF")
            if has_hyperparams and self.retriever.is_too_similar(config_dict, threshold=0.99):
                print("  [SKIP] 配置过于相似")
                if self.optuna_planner and "_optuna_trial_number" in config_dict:
                    self.optuna_planner.update_result(
                        config_dict["_optuna_trial_number"], 0.0
                    )
                round_num += 1
                continue

            # 6. 执行实验
            result = self.runner.run(config_dict)
            if result.status == "failed":
                print(f"  [ERROR] {result.error}")
                if self.optuna_planner and "_optuna_trial_number" in config_dict:
                    self.optuna_planner.update_result(
                        config_dict["_optuna_trial_number"], 0.0
                    )
                round_num += 1
                continue

            self.budget.record_round(result.train_time)

            # 7. 记录实验
            metric_key = "val_accuracy" if self.task_type == "classification" else "ndcg@k"
            improved = result.metric > self._best_metric

            record = ExperimentRecord(
                round_num=round_num,
                model_type=model_type,
                config=config_dict,
                metrics=result.metrics,
                elapsed_seconds=result.train_time,
                task=self.task_type,
                status="success",
            )
            self.memory.add(record)
            self.budget.record_improvement(improved)

            # 8. 更新最佳模型
            if improved:
                self._best_metric = result.metric
                self._best_model = result.model
                print(f"  ★ 新最佳! {metric_key}={result.metric:.4f}")

            # 9. 更新 Optuna
            if self.optuna_planner and "_optuna_trial_number" in config_dict:
                self.optuna_planner.update_result(
                    config_dict["_optuna_trial_number"], result.metric
                )

            # 10. 更新 TopK Pool
            if self.topk_pool:
                self.topk_pool.add(
                    config=config_dict,
                    metric=result.metric,
                    model_name=model_type,
                    train_time=result.train_time,
                    round_num=round_num,
                )

            # 11. 更新 Bandit (Compute-Aware Reward)
            self.bandit_planner.bandit.update_compute_aware(
                arm=model_type,
                metric=result.metric,
                best_metric=self._best_metric,
                runtime=result.train_time,
            )

            # 12. 添加到 Ensemble
            if result.model is not None:
                self.ensemble.add_model(
                    model=result.model,
                    config=config_dict,
                    metric=result.metric,
                )

            # 13. 更新 Retriever 和 KnowledgeBase
            self.retriever.index(self.memory)
            self.knowledge_base.add(KnowledgeEntry(
                model_name=model_type,
                task_type=self.task_type,
                best_metric=result.metric,
                best_config=config_dict,
                insights=[reflection_result.observation] if reflection_result else [],
            ))

            # 14. 提取实验模式
            patterns = self.pattern_extractor.extract(self.memory, self.task_type)
            if patterns:
                print(f"  模式: {patterns[0].description[:60]}...")

            # 15. 记录日志
            self.logger.log(
                round_num=round_num,
                config=config_dict,
                metrics=result.metrics,
                feedback={"trend": feedback["trend"], "suggestions": feedback["suggestions"]},
                strategy=decision.reason,
                elapsed_seconds=result.train_time,
                decision={
                    "bandit_arm": selected_arm,
                    "qwen_decision": decision.model_dump(),
                    "reflection": reflection_result.model_dump() if reflection_result else {},
                    "patterns": [p.description for p in patterns[:3]],
                    "topk_rank": self.topk_pool.entries[0].rank if self.topk_pool and self.topk_pool.entries else None,
                },
                runtime={"device": str(self.device), "elapsed_seconds": result.train_time},
            )

            print(f"  耗时: {result.train_time:.1f}s | {metric_key}: {result.metric:.4f}")
            print(f"  反思: {reflection_result.observation[:60]}...")
            if self.topk_pool:
                print(f"  TopK: {self.topk_pool.summary()}")
            print(self.budget.format_status())

            round_num += 1

        # 保存
        self.logger.save()
        if self.optuna_planner:
            self.optuna_planner.save()
        if self.topk_pool:
            self.topk_pool.save()
        self.memory.save(f"{self.output_dir}/memory_{self.task_type}.json")
        self.knowledge_base.save()

        print(f"\n[Agent] 实验结束，共 {round_num} 轮")
        print(self.memory.summary())

        # 构建集成模型
        self._build_ensemble()

        return {
            "best_metric": self._best_metric,
            "best_config": self.memory.get_best(self.task_type).config if self.memory.get_best(self.task_type) else {},
            "num_rounds": round_num,
        }

    def _generate_candidates(self, model_type: str) -> list[dict]:
        """生成候选配置。

        优先级：
        1. TopK Pool 中的配置（80%）
        2. Optuna 采样（20%）
        """
        candidates = []

        # 从 TopK Pool 生成候选
        if self.topk_pool and len(self.topk_pool) > 0:
            focus_configs = self.topk_pool.get_focus_configs(n=5, focus_ratio=0.8)
            candidates.extend(focus_configs)

        # 从 Optuna 生成候选
        if self.optuna_planner:
            self.optuna_planner.set_model_type(model_type)
            for _ in range(2):
                try:
                    cfg = self.optuna_planner.next_config()
                    cfg["model_type"] = model_type
                    candidates.append(cfg)
                except Exception:
                    break

        if not candidates:
            cfg = self.bandit_planner._default_config(model_type)
            candidates.append(cfg)

        return candidates[:5]

    def _build_ensemble(self):
        """构建集成模型。"""
        if len(self.ensemble) < 2:
            print("[Agent] 模型数量不足，跳过集成")
            return

        print(f"\n[Agent] 构建集成模型 ({len(self.ensemble)} 个模型)")
        self.ensemble.build(weights="metric")
        print(f"  集成权重: {[(m['config'].get('model_type', '?'), m['weight']) for m in self.ensemble.models]}")

    def generate_submission(self, output_path: str):
        """基于最佳模型或集成模型生成提交文件。"""
        print(f"\n[Agent] 生成提交文件: {output_path}")

        if self.task_type == "classification":
            self._generate_cls_submission(output_path)
        else:
            self._generate_rec_submission(output_path)

    def _generate_cls_submission(self, output_path: str):
        data = self._data
        from data_loader import classification_to_pyg
        pyg_data = classification_to_pyg(data, self.device)

        # 优先使用集成模型
        if len(self.ensemble) >= 2:
            print("  使用集成模型预测")
            predictions = self.ensemble.predict_cls(pyg_data, self.device)
        elif self._best_model is not None:
            print("  使用单一最佳模型预测")
            predictions = predict_gnn(self._best_model, pyg_data)
        else:
            print("[Agent] 没有可用的模型")
            return

        import pandas as pd
        df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions})
        df.to_csv(output_path, index=False)
        print(f"  提交文件已生成: {output_path} ({len(df)} 行)")

    def _generate_rec_submission(self, output_path: str):
        test_df = self._data["test_df"]
        results = []

        for _, row in test_df.iterrows():
            uid = row["uid"]
            seq = parse_seq_dedup(row["item_seq_dedup"])

            # 优先使用集成模型
            if len(self.ensemble) >= 2:
                pred_list = self.ensemble.predict_rec(uid, seq, top_k=10)
            elif self._best_model is not None:
                pred_list = self._best_model.predict(uid, seq, top_k=10)
            else:
                pred_list = []

            results.append({"uid": uid, "prediction": ",".join(pred_list)})

        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"  提交文件已生成: {output_path} ({len(df)} 行)")

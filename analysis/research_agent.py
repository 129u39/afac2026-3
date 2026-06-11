"""自主科研 Agent：闭环迭代优化系统。"""

import os
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Any


@dataclass
class ExperimentResult:
    """实验结果。"""
    config: dict
    metric: float
    metric_name: str
    train_time: float
    status: str = "success"
    error: str = ""
    timestamp: str = ""
    round_num: int = 0


@dataclass
class Diagnosis:
    """诊断结果。"""
    bottleneck: str
    hypotheses: list[str]
    priority: str = "medium"


class ResearchAgent:
    """自主科研 Agent。

    闭环迭代：
    Literature → Diagnosis → Design → Experiment → Memory → Decision

    决策：CONTINUE / PIVOT / STOP
    """

    def __init__(
        self,
        task_type: str,
        run_fn: Callable,
        llm_client=None,
        memory_path: str = "output/research_memory.json",
    ):
        """
        Args:
            task_type: "classification" 或 "recommendation"
            run_fn: 实验执行函数，接受 config dict，返回 ExperimentResult
            llm_client: LLM 客户端
            memory_path: 记忆存储路径
        """
        self.task_type = task_type
        self.run_fn = run_fn
        self.llm_client = llm_client
        self.memory_path = memory_path

        self.memory: list[ExperimentResult] = []
        self.hypotheses: list[str] = []
        self.budget_remaining: float = 7200
        self.round_num: int = 0
        self.max_rounds: int = 50

    def run(self, initial_config: dict) -> dict:
        """运行自主科研循环。"""
        self._load_memory()

        # 第1轮：冷启动
        print("\n" + "="*60)
        print("第1轮：冷启动")
        print("="*60)

        # Literature
        self._literature_phase(initial_config)

        # Design + Experiment
        result = self._experiment_phase(initial_config)
        self._memory_phase(result)

        # Diagnosis
        self._diagnosis_phase()

        # 第2~N轮：迭代优化
        for round_num in range(1, self.max_rounds):
            if self.budget_remaining < 60:
                print("\n预算耗尽，停止")
                break

            print(f"\n{'='*60}")
            print(f"第{round_num+1}轮：迭代优化")
            print(f"{'='*60}")

            # Diagnosis
            self._diagnosis_phase()

            # Design
            config = self._design_phase()

            # Experiment
            result = self._experiment_phase(config)
            self._memory_phase(result)

            # Decision
            decision = self._decision_phase()
            if decision == "STOP":
                break

        return self._summarize()

    def _literature_phase(self, config: dict):
        """文献解析阶段：读取数据、分析代码。"""
        print("[Literature] 分析数据和代码...")

        # 分析当前配置
        print(f"  任务类型: {self.task_type}")
        print(f"  初始配置: {config}")

        # 调用 LLM 分析（如果有）
        if self.llm_client and self.llm_client.available:
            prompt = f"""分析以下实验配置，提出初始假设：

任务: {self.task_type}
配置: {json.dumps(config, indent=2)}
历史实验: {len(self.memory)} 轮

请提出3个最有可能提升性能的假设。"""
            try:
                response = self.llm_client.chat(
                    "你是一个机器学习研究专家。", prompt
                )
                print(f"  LLM 建议: {response[:100]}...")
            except Exception as e:
                print(f"  LLM 调用失败: {e}")

    def _diagnosis_phase(self):
        """诊断阶段：分析历史，识别瓶颈。"""
        print("[Diagnosis] 分析历史记录...")

        if not self.memory:
            return

        # 分析最近实验
        recent = self.memory[-5:] if len(self.memory) >= 5 else self.memory
        metrics = [r.metric for r in recent]
        times = [r.train_time for r in recent]

        print(f"  最近{len(recent)}轮:")
        print(f"    指标: {metrics}")
        print(f"    耗时: {[f'{t:.1f}s' for t in times]}")

        # 识别趋势
        if len(metrics) >= 2:
            if metrics[-1] > metrics[0]:
                print(f"  趋势: 改善中 (+{metrics[-1]-metrics[0]:.4f})")
                self.hypotheses = [
                    "继续当前方向优化",
                    "尝试更大模型容量",
                    "调整学习率策略",
                ]
            elif metrics[-1] < metrics[0] - 0.01:
                print(f"  趋势: 下降 ({metrics[-1]-metrics[0]:.4f})")
                self.hypotheses = [
                    "回退到最佳配置",
                    "减少模型复杂度",
                    "增加正则化",
                ]
            else:
                print(f"  趋势: 平台期")
                self.hypotheses = [
                    "尝试新模型架构",
                    "特征工程",
                    "调整搜索空间",
                ]

        # 调用 LLM 诊断
        if self.llm_client and self.llm_client.available:
            prompt = f"""诊断实验状态：

最近{len(recent)}轮结果: {[f'{r.metric:.4f}' for r in recent]}
趋势: {self.hypotheses}

请分析瓶颈，提出下一步建议。"""
            try:
                response = self.llm_client.chat(
                    "你是一个机器学习研究专家。", prompt
                )
                print(f"  LLM 诊断: {response[:80]}...")
            except Exception:
                pass

    def _design_phase(self) -> dict:
        """设计阶段：生成新配置。"""
        print("[Design] 生成新配置...")

        if not self.memory:
            # 冷启动：使用默认配置
            return self._default_config()

        # 基于最近最佳配置微调
        best = max(self.memory, key=lambda r: r.metric)
        config = self._perturb_config(best.config)

        print(f"  基于最佳配置微调: {best.config.get('model_type', '?')} → {config.get('model_type', '?')}")
        return config

    def _experiment_phase(self, config: dict) -> ExperimentResult:
        """实验阶段：训练 + 评估。"""
        print(f"[Experiment] 训练 {config.get('model_type', 'unknown')}...")

        start_time = time.time()
        try:
            result = self.run_fn(config)
            result.train_time = time.time() - start_time
            result.timestamp = datetime.now().isoformat()
            result.round_num = self.round_num
            self.budget_remaining -= result.train_time
            return result
        except Exception as e:
            return ExperimentResult(
                config=config,
                metric=0.0,
                metric_name="error",
                train_time=time.time() - start_time,
                status="failed",
                error=str(e),
                timestamp=datetime.now().isoformat(),
                round_num=self.round_num,
            )

    def _memory_phase(self, result: ExperimentResult):
        """记忆阶段：保存结果。"""
        self.memory.append(result)
        self._save_memory()

        if result.status == "success":
            print(f"  结果: {result.metric_name}={result.metric:.4f}")
        else:
            print(f"  失败: {result.error[:50]}")

    def _decision_phase(self) -> str:
        """决策阶段：CONTINUE / PIVOT / STOP。"""
        print("[Decision] 决策中...")

        # 检查预算
        if self.budget_remaining < 60:
            print("  → STOP: 预算耗尽")
            return "STOP"

        # 检查改善
        if len(self.memory) >= 5:
            recent = self.memory[-5:]
            metrics = [r.metric for r in recent]
            if len(metrics) >= 2 and metrics[-1] <= metrics[0]:
                no_improve = sum(1 for i in range(1, len(metrics)) if metrics[i] <= metrics[i-1])
                if no_improve >= 4:
                    print("  → STOP: 连续无改善")
                    return "STOP"

        # 检查轮次
        if self.round_num >= self.max_rounds - 1:
            print("  → STOP: 达到最大轮次")
            return "STOP"

        print("  → CONTINUE")
        return "CONTINUE"

    def _default_config(self) -> dict:
        """默认配置。"""
        if self.task_type == "classification":
            return {
                "model_type": "Hybrid",
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
            }
        else:
            return {
                "model_type": "Popularity",
            }

    def _perturb_config(self, config: dict) -> dict:
        """微调配置。"""
        import random
        cfg = config.copy()

        if self.task_type == "classification":
            param = random.choice(["hidden_dim", "dropout", "lr", "num_layers", "weight_decay"])
            if param == "hidden_dim":
                cfg["hidden_dim"] = random.choice([64, 128, 256])
            elif param == "dropout":
                cfg["dropout"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
            elif param == "lr":
                cfg["lr"] = random.choice([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
            elif param == "num_layers":
                cfg["num_layers"] = random.choice([2, 3, 4])
            elif param == "weight_decay":
                cfg["weight_decay"] = random.choice([0.0, 1e-5, 5e-5, 1e-4, 5e-4])
        else:
            if "embedding_dim" in cfg:
                cfg["embedding_dim"] = random.choice([32, 64, 128])

        return cfg

    def _save_memory(self):
        """保存记忆。"""
        data = {
            "task_type": self.task_type,
            "round_num": self.round_num,
            "budget_remaining": self.budget_remaining,
            "experiments": [asdict(r) for r in self.memory],
        }
        os.makedirs(os.path.dirname(self.memory_path) if os.path.dirname(self.memory_path) else ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_memory(self):
        """加载记忆。"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory = [ExperimentResult(**r) for r in data.get("experiments", [])]
                self.round_num = data.get("round_num", 0)
                self.budget_remaining = data.get("budget_remaining", 7200)
                print(f"  加载记忆: {len(self.memory)} 轮历史")
            except Exception:
                pass

    def _summarize(self) -> dict:
        """总结实验结果。"""
        if not self.memory:
            return {"best_metric": 0.0, "num_rounds": 0}

        successful = [r for r in self.memory if r.status == "success"]
        if not successful:
            return {"best_metric": 0.0, "num_rounds": len(self.memory)}

        best = max(successful, key=lambda r: r.metric)
        return {
            "best_metric": best.metric,
            "best_config": best.config,
            "num_rounds": len(self.memory),
            "best_round": best.round_num,
        }

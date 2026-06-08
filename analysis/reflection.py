"""反思 Agent：分析实验结果，生成反思与下一步行动建议。"""

from memory import ExperimentMemory, ExperimentRecord


class ReflectionAgent:
    """实验反思 Agent。

    输入：最近的实验记录 + 反馈分析结果
    输出：观察、推理、下一步行动、置信度

    参与 planner.next_config() 决策。
    """

    def reflect(
        self,
        memory: ExperimentMemory,
        feedback: dict,
        task_type: str,
    ) -> dict:
        """生成实验反思。

        Args:
            memory: 实验记忆
            feedback: FeedbackAnalyzer 的输出
            task_type: "classification" 或 "recommendation"

        返回:
            {
                "observation": str,   # 观察到了什么
                "reasoning": str,     # 为什么会出现这种现象
                "next_action": str,   # 建议下一步做什么
                "confidence": float,  # 对建议的置信度 (0~1)
            }
        """
        if not memory.records:
            return {
                "observation": "尚无实验记录",
                "reasoning": "需要先运行基线实验获取初始指标",
                "next_action": "run_baseline",
                "confidence": 1.0,
            }

        metric_key = feedback.get("metric_key", "val_accuracy")
        trend = feedback.get("trend", "unknown")
        risk = feedback.get("risk", "none")
        best_config = feedback.get("best_config")
        recent = memory.recent(5)

        # ── 观察 ──────────────────────────────────────

        observation = self._make_observation(recent, metric_key, trend, risk)

        # ── 推理 ──────────────────────────────────────

        reasoning = self._make_reasoning(recent, metric_key, trend, risk, best_config)

        # ── 行动建议 ──────────────────────────────────

        next_action, confidence = self._suggest_action(
            memory, feedback, task_type, trend, risk
        )

        return {
            "observation": observation,
            "reasoning": reasoning,
            "next_action": next_action,
            "confidence": confidence,
        }

    def _make_observation(
        self,
        recent: list[ExperimentRecord],
        metric_key: str,
        trend: str,
        risk: str,
    ) -> str:
        """生成观察描述。"""
        if len(recent) < 2:
            return f"只有 {len(recent)} 轮实验数据，信息不足"

        metrics = [r.metrics.get(metric_key, 0) for r in recent]
        best_metric = max(metrics)
        last_metric = metrics[-1]
        models = [r.model_type for r in recent]

        parts = [
            f"最近 {len(recent)} 轮实验，最佳 {metric_key}={best_metric:.4f}，"
            f"最新 {metric_key}={last_metric:.4f}。",
            f"尝试的模型: {', '.join(set(models))}。",
            f"趋势: {trend}，风险: {risk}。",
        ]

        if trend == "overfitting":
            parts.append("指标在达到峰值后开始下降，可能过拟合。")
        elif trend == "unstable":
            parts.append("指标波动较大，训练不稳定。")
        elif trend == "plateau":
            parts.append("指标停滞不前，需要新的策略突破。")

        return " ".join(parts)

    def _make_reasoning(
        self,
        recent: list[ExperimentRecord],
        metric_key: str,
        trend: str,
        risk: str,
        best_config: dict | None,
    ) -> str:
        """生成推理描述。"""
        parts = []

        if trend == "improving":
            parts.append("当前方向有效，模型正在学习有意义的特征。")
            if best_config:
                parts.append(f"最佳配置 ({best_config.get('model_type', '?')}) 表现良好，值得继续微调。")

        elif trend == "overfitting":
            parts.append("模型可能过于复杂或训练时间过长。")
            parts.append("验证集性能下降表明泛化能力不足。")
            parts.append("应增加正则化或减少模型容量。")

        elif trend == "unstable":
            parts.append("学习率可能过大，导致优化过程震荡。")
            parts.append("或者模型架构对初始化敏感。")
            parts.append("应降低学习率或使用更稳定的优化器设置。")

        elif trend == "plateau":
            parts.append("当前模型/超参组合可能已达到表达能力上限。")
            parts.append("需要尝试不同的模型架构或更大的搜索空间。")

        elif trend == "declining":
            parts.append("最近的修改可能引入了负面影响。")
            parts.append("应考虑回退到之前的最佳配置。")

        else:
            parts.append("数据不足，无法做出可靠推理。")

        return " ".join(parts)

    def _suggest_action(
        self,
        memory: ExperimentMemory,
        feedback: dict,
        task_type: str,
        trend: str,
        risk: str,
    ) -> tuple[str, float]:
        """建议下一步行动。

        返回:
            (action, confidence)
        """
        tried_models = feedback.get("tried_models", [])

        if task_type == "classification":
            all_models = {"GCN", "GAT", "GraphSAGE"}
        else:
            all_models = {"Popularity", "ItemCF", "BPR_MF", "SASRec", "LightGCN"}

        untried = all_models - set(tried_models)

        if trend == "improving":
            return "fine_tune_best", 0.8

        if trend == "overfitting":
            return "increase_regularization", 0.9

        if trend == "unstable":
            return "reduce_learning_rate", 0.85

        if trend == "plateau":
            if untried:
                return "try_new_architecture", 0.7
            return "perturb_best_config", 0.6

        if trend == "declining":
            return "revert_to_best", 0.8

        # insufficient_data
        return "run_baseline", 1.0

    def get_decision_context(self) -> dict:
        """返回最近一次反思的决策上下文（用于日志记录）。"""
        return {
            "reflection_used": True,
            "last_reflection": getattr(self, "_last_reflection", None),
        }

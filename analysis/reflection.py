"""反思 Agent V2：规则式 + Qwen LLM 双模式。"""

from memory import ExperimentMemory, ExperimentRecord
from llm.schemas import ReflectionResult


class ReflectionAgent:
    """实验反思 Agent V2。

    支持两种模式：
    1. Qwen 模式：调用 LLM 生成深度反思（需要 API Key）
    2. 规则模式：基于规则的快速反思（fallback）

    输入：最近的实验记录 + 反馈分析结果
    输出：ReflectionResult（观察、推理、行动、置信度）
    """

    def __init__(self, qwen_client=None):
        """
        Args:
            qwen_client: QwenClient 实例，None 时使用规则模式
        """
        self.qwen_client = qwen_client
        self._last_reflection: ReflectionResult | None = None

    def reflect(
        self,
        memory: ExperimentMemory,
        feedback: dict,
        task_type: str,
        similar_experiments: list[dict] | None = None,
        budget_state: dict | None = None,
    ) -> ReflectionResult:
        """生成实验反思。

        Args:
            memory: 实验记忆
            feedback: FeedbackAnalyzer 的输出
            task_type: "classification" 或 "recommendation"
            similar_experiments: 相似历史实验（来自 Retriever）
            budget_state: 预算状态

        返回:
            ReflectionResult
        """
        # 尝试 Qwen 模式
        if self.qwen_client and self.qwen_client.available:
            try:
                result = self._qwen_reflect(
                    memory, feedback, task_type, similar_experiments, budget_state
                )
                if result:
                    self._last_reflection = result
                    return result
            except Exception:
                pass  # fallback to rule-based

        # 规则模式 fallback
        result = self._rule_based_reflect(memory, feedback, task_type)
        self._last_reflection = result
        return result

    def _qwen_reflect(
        self,
        memory: ExperimentMemory,
        feedback: dict,
        task_type: str,
        similar_experiments: list[dict] | None,
        budget_state: dict | None,
    ) -> ReflectionResult | None:
        """调用 Qwen 生成反思。"""
        from llm.prompts import SYSTEM_PROMPT, REFLECTION_PROMPT
        from llm.parser import parse_structured

        # 组装上下文
        recent = memory.recent(5)
        recent_str = self._format_experiments(recent)

        similar_str = "无"
        if similar_experiments:
            similar_str = "\n".join(
                f"- {s.get('model_type', '?')}: metric={s.get('metrics', {}).get('val_accuracy', s.get('metrics', {}).get('ndcg@k', '?'))}, similarity={s.get('similarity', 0):.2f}"
                for s in similar_experiments[:5]
            )

        best = memory.get_best(task_type)
        best_str = self._format_one(best) if best else "无"

        budget_str = f"已用 {budget_state.get('elapsed_seconds', 0)/60:.1f}min, 剩余 {budget_state.get('remaining_seconds', 0)/60:.1f}min" if budget_state else "未知"

        feedback_str = f"趋势: {feedback.get('trend', '?')}, 风险: {feedback.get('risk', '?')}, 建议: {', '.join(feedback.get('suggestions', [])[:3])}"

        user_prompt = REFLECTION_PROMPT.format(
            task_type=task_type,
            recent_experiments=recent_str,
            similar_experiments=similar_str,
            best_experiment=best_str,
            budget_state=budget_str,
            feedback=feedback_str,
        )

        response = self.qwen_client.chat(SYSTEM_PROMPT, user_prompt)
        return parse_structured(response, ReflectionResult)

    def _rule_based_reflect(
        self,
        memory: ExperimentMemory,
        feedback: dict,
        task_type: str,
    ) -> ReflectionResult:
        """规则式反思（V1 逻辑）。"""
        if not memory.records:
            return ReflectionResult(
                observation="尚无实验记录",
                reasoning="需要先运行基线实验获取初始指标",
                next_action="run_baseline",
                confidence=1.0,
            )

        metric_key = feedback.get("metric_key", "val_accuracy")
        trend = feedback.get("trend", "unknown")
        risk = feedback.get("risk", "none")
        recent = memory.recent(5)

        # 观察
        observation = self._make_observation(recent, metric_key, trend, risk)

        # 推理
        reasoning = self._make_reasoning(trend, risk, feedback.get("best_config"))

        # 行动
        next_action, confidence = self._suggest_action(memory, feedback, task_type, trend, risk)

        return ReflectionResult(
            observation=observation,
            reasoning=reasoning,
            next_action=next_action,
            confidence=confidence,
        )

    def _make_observation(self, recent, metric_key, trend, risk):
        metrics = [r.metrics.get(metric_key, 0) for r in recent]
        best_metric = max(metrics) if metrics else 0
        last_metric = metrics[-1] if metrics else 0
        models = list({r.model_type for r in recent})

        parts = [
            f"最近 {len(recent)} 轮，最佳 {metric_key}={best_metric:.4f}，最新={last_metric:.4f}。",
            f"模型: {', '.join(models)}。趋势: {trend}，风险: {risk}。",
        ]
        return " ".join(parts)

    def _make_reasoning(self, trend, risk, best_config):
        if trend == "improving":
            return "当前方向有效，模型正在学习有意义的特征。"
        if trend == "overfitting":
            return "模型可能过于复杂或训练时间过长，应增加正则化。"
        if trend == "unstable":
            return "学习率可能过大，应降低学习率或使用更稳定的设置。"
        if trend == "plateau":
            return "当前模型/超参组合可能已达到表达能力上限，需要新策略。"
        if trend == "declining":
            return "最近的修改可能引入了负面影响，应考虑回退。"
        return "数据不足，无法做出可靠推理。"

    def _suggest_action(self, memory, feedback, task_type, trend, risk):
        tried_models = feedback.get("tried_models", [])

        if task_type == "classification":
            all_models = {"GCN", "GAT", "GraphSAGE"}
        else:
            all_models = {"Popularity", "ItemCF", "BPR_MF", "LightGCN", "SASRec"}

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
        return "run_baseline", 1.0

    def _format_experiments(self, records: list[ExperimentRecord]) -> str:
        if not records:
            return "无"
        lines = []
        for r in records:
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in r.metrics.items()
                if isinstance(v, (int, float))
            )
            lines.append(f"- Round {r.round_num}: {r.model_type} | {metric_str} | {r.elapsed_seconds:.1f}s")
        return "\n".join(lines)

    def _format_one(self, rec: ExperimentRecord) -> str:
        metric_str = ", ".join(
            f"{k}={v:.4f}" for k, v in rec.metrics.items()
            if isinstance(v, (int, float))
        )
        return f"{rec.model_type} | {metric_str} | config={rec.config}"

    def get_last_reflection(self) -> ReflectionResult | None:
        return self._last_reflection

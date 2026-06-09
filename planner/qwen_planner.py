"""Qwen Planner：LLM 驱动的最终决策器。"""

from llm.schemas import PlannerDecision


class QwenPlanner:
    """Qwen 驱动的实验规划器。

    职责：在 Bandit + Optuna 提供的候选配置中做最终选择。
    不直接生成参数，而是在候选中选择最合适的。

    与 BanditPlanner 和 OptunaPlanner 配合：
    1. BanditPlanner 选择模型架构
    2. OptunaPlanner 生成 Top-K 候选配置
    3. QwenPlanner 在候选中做最终决策
    """

    def __init__(self, qwen_client=None):
        """
        Args:
            qwen_client: QwenClient 实例，None 时使用简单选择策略
        """
        self.qwen_client = qwen_client

    def select(
        self,
        candidate_configs: list[dict],
        reflection: dict | None = None,
        budget_state: dict | None = None,
        history_summary: str = "",
        best_experiment: dict | None = None,
        task_type: str = "classification",
    ) -> PlannerDecision:
        """从候选配置中选择最优配置。

        Args:
            candidate_configs: 候选配置列表（来自 Optuna）
            reflection: 反思结果（来自 ReflectionAgent）
            budget_state: 预算状态
            history_summary: 实验历史摘要
            best_experiment: 最佳实验信息
            task_type: 任务类型

        返回:
            PlannerDecision
        """
        if not candidate_configs:
            # 无候选配置，返回空决策
            return PlannerDecision(
                selected_config={},
                reason="无候选配置",
                confidence=0.0,
            )

        # 只有一个候选，直接选择
        if len(candidate_configs) == 1:
            return PlannerDecision(
                selected_config=candidate_configs[0],
                reason="唯一候选配置",
                confidence=0.8,
            )

        # 尝试 Qwen 模式
        if self.qwen_client and self.qwen_client.available:
            try:
                result = self._qwen_select(
                    candidate_configs, reflection, budget_state,
                    history_summary, best_experiment, task_type,
                )
                if result:
                    return result
            except Exception:
                pass  # fallback

        # 简单策略 fallback：选择第一个候选
        return PlannerDecision(
            selected_config=candidate_configs[0],
            reason="默认选择第一个候选配置",
            confidence=0.5,
        )

    def _qwen_select(
        self,
        candidate_configs: list[dict],
        reflection: dict | None,
        budget_state: dict | None,
        history_summary: str,
        best_experiment: dict | None,
        task_type: str,
    ) -> PlannerDecision | None:
        """调用 Qwen 做选择。"""
        from llm.prompts import SYSTEM_PROMPT, PLANNER_PROMPT
        from llm.parser import parse_structured
        import json

        # 格式化候选配置
        configs_str = json.dumps(candidate_configs, ensure_ascii=False, indent=2)

        # 格式化反思
        reflection_str = "无"
        if reflection:
            reflection_str = json.dumps(reflection, ensure_ascii=False, indent=2)

        # 格式化预算
        budget_str = "未知"
        if budget_state:
            budget_str = f"已用 {budget_state.get('elapsed_seconds', 0)/60:.1f}min, 剩余 {budget_state.get('remaining_seconds', 0)/60:.1f}min"

        # 格式化最佳实验
        best_str = "无"
        if best_experiment:
            best_str = json.dumps(best_experiment, ensure_ascii=False, indent=2)

        user_prompt = PLANNER_PROMPT.format(
            task_type=task_type,
            candidate_configs=configs_str,
            reflection=reflection_str,
            budget_state=budget_str,
            history_summary=history_summary or "无",
            best_experiment=best_str,
        )

        response = self.qwen_client.chat(SYSTEM_PROMPT, user_prompt)
        return parse_structured(response, PlannerDecision)

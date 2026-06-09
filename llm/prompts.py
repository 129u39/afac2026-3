"""LLM 提示词模板。"""

# ── 系统提示词 ──────────────────────────────────────

SYSTEM_PROMPT = """你是一个自动化机器学习实验 Agent。你的职责是分析实验结果、反思原因、并做出下一步决策。

你需要：
1. 观察实验数据，识别趋势和模式
2. 分析成功和失败的原因
3. 基于证据做出决策，而非猜测
4. 考虑时间预算限制

输出必须是有效的 JSON 格式。"""

# ── 反思提示词 ──────────────────────────────────────

REFLECTION_PROMPT = """## 任务
分析最近的实验结果，生成反思和下一步行动建议。

## 任务类型
{task_type}

## 最近实验
{recent_experiments}

## 相似历史实验
{similar_experiments}

## 最佳实验
{best_experiment}

## 预算状态
{budget_state}

## 反馈分析
{feedback}

## 要求
请以 JSON 格式输出：
{{
    "observation": "观察到了什么",
    "reasoning": "为什么会出现这种现象",
    "next_action": "下一步行动（fine_tune_best / increase_regularization / try_new_architecture / reduce_learning_rate / revert_to_best / run_baseline）",
    "confidence": 0.0到1.0的置信度
}}"""

# ── 规划提示词 ──────────────────────────────────────

PLANNER_PROMPT = """## 任务
从候选配置中选择最优的实验配置。

## 任务类型
{task_type}

## 候选配置
{candidate_configs}

## 最近反思
{reflection}

## 预算状态
{budget_state}

## 实验历史摘要
{history_summary}

## 最佳实验
{best_experiment}

## 要求
请以 JSON 格式输出：
{{
    "selected_config": {{选中的配置}},
    "reason": "选择原因",
    "confidence": 0.0到1.0的置信度
}}"""

# ── 停止判断提示词 ──────────────────────────────────

STOP_PROMPT = """## 任务
判断是否应该停止实验。

## 实验历史
{history}

## 预算状态
{budget_state}

## 最近趋势
{trend}

## 要求
请以 JSON 格式输出：
{{
    "should_stop": true或false,
    "reason": "停止或继续的原因"
}}"""

# ── 知识提取提示词 ──────────────────────────────────

KNOWLEDGE_PROMPT = """## 任务
从实验结果中提取可迁移的知识。

## 实验结果
{experiment_result}

## 任务类型
{task_type}

## 要求
请以 JSON 格式输出：
{{
    "findings": ["发现1", "发现2"],
    "suggestions": ["建议1", "建议2"],
    "risk_assessment": "风险评估"
}}"""

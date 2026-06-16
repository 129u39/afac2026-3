"""LLM 结构化输出 Schema：Pydantic 模型定义。"""

from pydantic import BaseModel, Field


class ReflectionResult(BaseModel):
    """反思 Agent 输出。"""

    observation: str = Field(
        description="观察：从最近实验中发现了什么"
    )
    reasoning: str = Field(
        description="推理：为什么会出现这种现象"
    )
    next_action: str = Field(
        description="下一步行动建议",
        examples=["fine_tune_best", "increase_regularization", "try_new_architecture",
                  "reduce_learning_rate", "revert_to_best", "run_baseline"]
    )
    confidence: float = Field(
        description="对建议的置信度 (0~1)",
        ge=0.0,
        le=1.0,
    )


class PlannerDecision(BaseModel):
    """Qwen Planner 输出。"""

    selected_config: dict = Field(
        description="选中的实验配置"
    )
    reason: str = Field(
        description="选择这个配置的原因"
    )
    confidence: float = Field(
        description="对这个选择的置信度 (0~1)",
        ge=0.0,
        le=1.0,
    )


class StopDecision(BaseModel):
    """停止判断输出。"""

    should_stop: bool = Field(
        description="是否应该停止实验"
    )
    reason: str = Field(
        description="停止或继续的原因"
    )


class ResearchResult(BaseModel):
    """研究分析输出。"""

    findings: list[str] = Field(
        description="发现的关键信息",
        default_factory=list,
    )
    suggestions: list[str] = Field(
        description="建议的下一步操作",
        default_factory=list,
    )
    risk_assessment: str = Field(
        description="风险评估",
        default="none",
    )


class HPOSuggestion(BaseModel):
    """LLM 驱动的超参数优化建议。"""

    reasoning: str = Field(
        description="基于实验历史的分析推理，说明为什么要调整这些参数"
    )
    config_updates: dict = Field(
        description="要修改的超参数及其新值（只包含要更新的参数）",
        examples=[{"learning_rate": 0.01, "n_estimators": 500}],
    )
    unchanged_params: list[str] = Field(
        description="建议保持不变的参数列表",
        default_factory=list,
    )
    confidence: float = Field(
        description="对此建议的置信度 (0~1)",
        ge=0.0,
        le=1.0,
    )

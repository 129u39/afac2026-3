"""预算管理器：跟踪时间消耗，保护超时，支持早停决策。V1 — 增强决策 API。"""

import time


class BudgetManager:
    """管理实验预算（时间），支持早停判断。"""

    def __init__(
        self,
        total_seconds: int = 7200,
        safety_margin: int = 300,
        max_no_improve_rounds: int = 10,
        min_remaining_seconds: int = 600,
    ):
        """
        Args:
            total_seconds: 总预算（秒）
            safety_margin: 安全余量（秒），预留用于生成提交
            max_no_improve_rounds: 连续无提升轮数上限
            min_remaining_seconds: 最低剩余时间阈值（秒）
        """
        self.total_seconds = total_seconds
        self.safety_margin = safety_margin
        self.effective_budget = total_seconds - safety_margin
        self.max_no_improve_rounds = max_no_improve_rounds
        self.min_remaining_seconds = min_remaining_seconds

        self.start_time = time.time()
        self.round_times: list[float] = []
        self._no_improve_rounds: int = 0

    # ── 时间追踪 ──────────────────────────────────────

    def elapsed(self) -> float:
        """已用时间（秒）。"""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """剩余有效时间（秒）。"""
        return max(0, self.effective_budget - self.elapsed())

    def avg_round_time(self) -> float:
        """平均单轮耗时。"""
        if not self.round_times:
            return 120.0  # 默认预估2分钟
        return sum(self.round_times) / len(self.round_times)

    def record_round(self, seconds: float):
        """记录一轮实验的耗时。"""
        self.round_times.append(seconds)

    # ── V1: 改进追踪 ─────────────────────────────────

    def record_improvement(self, improved: bool):
        """记录本轮是否有提升，更新无提升计数。

        Args:
            improved: 本轮是否产生了指标提升
        """
        if improved:
            self._no_improve_rounds = 0
        else:
            self._no_improve_rounds += 1

    @property
    def no_improvement_rounds(self) -> int:
        """连续无提升的轮数。"""
        return self._no_improve_rounds

    # ── V1: 决策 API ─────────────────────────────────

    def has_time(self, estimated_round_seconds: float = 0) -> bool:
        """是否有足够时间再跑一轮实验。

        Args:
            estimated_round_seconds: 预估下一轮耗时，0则使用平均值
        """
        if estimated_round_seconds <= 0:
            estimated_round_seconds = self.avg_round_time()
        # 至少需要60秒才尝试
        return self.remaining() > max(estimated_round_seconds, 60)

    def can_run(self, config: dict) -> bool:
        """根据配置预估，判断是否有足够时间完成一轮实验。

        根据模型类型估算耗时:
        - Popularity: ~1s
        - ItemCF: ~40s
        - BPR_MF: ~10s
        - SASRec: ~80s
        - GCN/GAT/GraphSAGE: ~30s

        Args:
            config: 实验配置字典
        """
        model_type = config.get("model_type", "")
        time_estimates = {
            "Popularity": 5,
            "ItemCF": 60,
            "BPR_MF": 30,
            "SASRec": 100,
            "LightGCN": 60,
            "GCN": 30,
            "GAT": 40,
            "GraphSAGE": 30,
        }
        estimated = time_estimates.get(model_type, self.avg_round_time())
        return self.remaining() > estimated + 60  # 预留60秒缓冲

    def should_continue(self) -> bool:
        """综合判断是否应继续实验。

        停止条件（满足任一即停止）:
        1. 连续无提升轮数 >= max_no_improve_rounds
        2. 剩余时间 < min_remaining_seconds
        3. 无足够时间再跑一轮
        """
        # 条件1: 连续无提升
        if self._no_improve_rounds >= self.max_no_improve_rounds:
            return False

        # 条件2: 剩余时间不足
        if self.remaining() < self.min_remaining_seconds:
            return False

        # 条件3: 无法再跑一轮
        if not self.has_time():
            return False

        return True

    def remaining_budget(self) -> dict:
        """返回结构化预算信息。"""
        return {
            "elapsed_seconds": self.elapsed(),
            "remaining_seconds": self.remaining(),
            "total_seconds": self.total_seconds,
            "effective_budget": self.effective_budget,
            "rounds_completed": len(self.round_times),
            "avg_round_time": self.avg_round_time(),
            "no_improvement_rounds": self._no_improve_rounds,
            "max_no_improve_rounds": self.max_no_improve_rounds,
            "should_continue": self.should_continue(),
        }

    # ── 状态输出 ──────────────────────────────────────

    def status(self) -> dict:
        """返回预算状态（兼容旧 API）。"""
        return self.remaining_budget()

    def format_status(self) -> str:
        """格式化状态字符串。"""
        s = self.status()
        elapsed_min = s["elapsed_seconds"] / 60
        remain_min = s["remaining_seconds"] / 60
        return (
            f"[Budget] 已用 {elapsed_min:.1f}min / "
            f"剩余 {remain_min:.1f}min / "
            f"已完成 {s['rounds_completed']} 轮 / "
            f"平均轮耗时 {s['avg_round_time']:.0f}s / "
            f"无提升 {s['no_improvement_rounds']}轮"
        )

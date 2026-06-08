"""预算管理器：跟踪时间消耗，保护超时。"""

import time


class BudgetManager:
    """管理实验预算（时间）。"""

    def __init__(self, total_seconds: int = 7200, safety_margin: int = 300):
        """
        Args:
            total_seconds: 总预算（秒）
            safety_margin: 安全余量（秒），预留用于生成提交
        """
        self.total_seconds = total_seconds
        self.safety_margin = safety_margin
        self.effective_budget = total_seconds - safety_margin
        self.start_time = time.time()
        self.round_times: list[float] = []

    def elapsed(self) -> float:
        """已用时间（秒）。"""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """剩余有效时间（秒）。"""
        return max(0, self.effective_budget - self.elapsed())

    def has_time(self, estimated_round_seconds: float = 0) -> bool:
        """是否有足够时间再跑一轮实验。

        Args:
            estimated_round_seconds: 预估下一轮耗时，0则使用平均值
        """
        if estimated_round_seconds <= 0:
            estimated_round_seconds = self.avg_round_time()
        # 至少需要60秒才尝试
        return self.remaining() > max(estimated_round_seconds, 60)

    def record_round(self, seconds: float):
        """记录一轮实验的耗时。"""
        self.round_times.append(seconds)

    def avg_round_time(self) -> float:
        """平均单轮耗时。"""
        if not self.round_times:
            return 120.0  # 默认预估2分钟
        return sum(self.round_times) / len(self.round_times)

    def status(self) -> dict:
        """返回预算状态。"""
        return {
            "elapsed": self.elapsed(),
            "remaining": self.remaining(),
            "total": self.total_seconds,
            "effective_budget": self.effective_budget,
            "rounds_completed": len(self.round_times),
            "avg_round_time": self.avg_round_time(),
        }

    def format_status(self) -> str:
        """格式化状态字符串。"""
        s = self.status()
        elapsed_min = s["elapsed"] / 60
        remain_min = s["remaining"] / 60
        return (
            f"[Budget] 已用 {elapsed_min:.1f}min / "
            f"剩余 {remain_min:.1f}min / "
            f"已完成 {s['rounds_completed']} 轮 / "
            f"平均轮耗时 {s['avg_round_time']:.0f}s"
        )

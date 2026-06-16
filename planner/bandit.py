"""UCB Bandit：用于模型架构选择的 Exploration vs Exploitation 策略。"""

import math
from dataclasses import dataclass, field


@dataclass
class ArmStats:
    """单个臂的统计信息。"""
    name: str
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls


class UCBBandit:
    """Upper Confidence Bound 多臂老虎机。

    UCB 公式: mean_reward + c * sqrt(log(t) / n)
    - mean_reward: 该臂的平均奖励（exploitation）
    - c * sqrt(log(t) / n): 探索项（exploration）
      - t: 总拉杆次数
      - n: 该臂被拉次数
      - c: 探索系数（越大越倾向探索）
    """

    def __init__(self, arms: list[str], c: float = 1.5):
        """
        Args:
            arms: 臂名称列表
            c: 探索系数
        """
        self.arms = {name: ArmStats(name=name) for name in arms}
        self.c = c
        self.total_pulls: int = 0

    def select_arm(self) -> str:
        """根据 UCB 公式选择最优臂。

        如果有未探索的臂，优先选择（确保每个臂至少被拉一次）。
        """
        # 优先选择未探索的臂
        for name, stats in self.arms.items():
            if stats.pulls == 0:
                return name

        # 计算 UCB 分数
        best_arm = None
        best_score = float("-inf")

        for name, stats in self.arms.items():
            exploitation = stats.mean_reward
            exploration = self.c * math.sqrt(
                math.log(self.total_pulls) / stats.pulls
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_arm = name

        return best_arm

    def update(self, arm: str, reward: float):
        """更新臂的统计信息。

        Args:
            arm: 臂名称
            reward: 奖励值（0~1 之间的指标值）
        """
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")
        self.arms[arm].pulls += 1
        self.arms[arm].total_reward += reward
        self.total_pulls += 1

    def update_compute_aware(self, arm: str, metric: float, best_metric: float, runtime: float):
        """Compute-Aware 奖励更新。

        reward = (metric - best_metric) / runtime
        优先选择快速提升的配置。

        Args:
            arm: 臂名称
            metric: 当前指标
            best_metric: 历史最佳指标
            runtime: 运行时间（秒）
        """
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")

        # 计算 compute-aware reward (归一化奖励)
        # improvement_ratio: (metric - best_metric) / max(best_metric, 1e-6)
        # speed_score: 1 / runtime
        # reward = 0.7 * improvement_ratio + 0.3 * speed_score
        if runtime > 0:
            improvement = max(0, metric - best_metric)
            improvement_ratio = improvement / max(best_metric, 1e-6)
            speed_score = 1.0 / max(runtime, 1e-6)
            reward = 0.7 * improvement_ratio + 0.3 * speed_score
        else:
            reward = 0.0

        self.arms[arm].pulls += 1
        self.arms[arm].total_reward += reward
        self.total_pulls += 1

    def get_stats(self) -> dict:
        """返回所有臂的统计信息。"""
        return {
            name: {
                "pulls": stats.pulls,
                "mean_reward": stats.mean_reward,
                "total_reward": stats.total_reward,
            }
            for name, stats in self.arms.items()
        }

    def get_arm_names(self) -> list[str]:
        """返回所有臂名称。"""
        return list(self.arms.keys())

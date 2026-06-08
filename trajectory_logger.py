"""过程日志记录器：记录实验轨迹，用于 B 榜提交。V1 — 增强决策与运行时记录。"""

import json
import os
from datetime import datetime


class TrajectoryLogger:
    """记录 Agent 的实验决策过程，满足 AFAC B 榜合规要求。"""

    def __init__(self, output_dir: str = "output", task_name: str = "task"):
        self.entries: list[dict] = []
        self.output_dir = output_dir
        self.task_name = task_name
        os.makedirs(output_dir, exist_ok=True)

    def log(
        self,
        round_num: int,
        config: dict,
        metrics: dict,
        feedback: dict,
        strategy: str = "",
        elapsed_seconds: float = 0,
        decision: dict | None = None,
        runtime: dict | None = None,
    ):
        """记录一轮实验。

        Args:
            round_num: 轮次号
            config: 实验配置
            metrics: 评估指标
            feedback: 反馈分析结果
            strategy: 策略说明
            elapsed_seconds: 本轮耗时
            decision: 决策详情（bandit/optuna 选择依据）
            runtime: 运行时信息（设备、内存等）
        """
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "feedback": feedback,
            "strategy": strategy,
            "elapsed_seconds": elapsed_seconds,
        }

        # V1 新增字段
        if decision is not None:
            entry["decision"] = decision
        else:
            entry["decision"] = {}

        if runtime is not None:
            entry["runtime"] = runtime
        else:
            entry["runtime"] = {"elapsed_seconds": elapsed_seconds}

        self.entries.append(entry)

    def log_decision(self, round_num: int, decision: dict):
        """记录决策详情（可独立调用，补充到已有条目）。

        Args:
            round_num: 轮次号
            decision: 决策字典，包含 bandit_arm, optuna_trial, reflection 等
        """
        for entry in self.entries:
            if entry["round"] == round_num:
                entry["decision"] = decision
                return
        # 如果找不到对应轮次，创建一个独立条目
        self.entries.append({
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "config": {},
            "metrics": {},
            "feedback": {},
            "strategy": "",
            "elapsed_seconds": 0,
            "runtime": {},
        })

    def save(self, filename: str | None = None):
        """保存日志到文件。"""
        if filename is None:
            filename = f"trajectory_{self.task_name}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        return path

    def summary(self) -> str:
        """生成日志摘要。"""
        lines = [f"共记录 {len(self.entries)} 轮实验"]
        for entry in self.entries:
            r = entry["round"]
            model = entry["config"].get("model_type", "unknown")
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in entry["metrics"].items()
                if isinstance(v, (int, float))
            )
            decision = entry.get("decision", {})
            decision_str = ""
            if decision:
                parts = []
                if "bandit_arm" in decision:
                    parts.append(f"bandit={decision['bandit_arm']}")
                if "optuna_trial" in decision:
                    parts.append(f"trial={decision['optuna_trial']}")
                if parts:
                    decision_str = f" [{', '.join(parts)}]"
            lines.append(f"  Round {r}: {model} | {metric_str}{decision_str}")
        return "\n".join(lines)

"""过程日志记录器：记录实验轨迹，用于 B 榜提交。"""

import json
import os
from datetime import datetime


class TrajectoryLogger:
    """记录 Agent 的实验决策过程。"""

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
    ):
        """记录一轮实验。"""
        entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "feedback": feedback,
            "strategy": strategy,
            "elapsed_seconds": elapsed_seconds,
        }
        self.entries.append(entry)

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
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in entry["metrics"].items() if isinstance(v, (int, float)))
            lines.append(f"  Round {r}: {model} | {metric_str}")
        return "\n".join(lines)

"""排行榜输出：实时记录最佳模型表现。"""
import json
import os
import csv
from datetime import datetime


class Leaderboard:
    """实时排行榜，记录最佳模型表现。

    自动保存 Top-10 实验结果，同时输出 JSON 和 CSV 格式。
    """

    CSV_COLUMNS = ["timestamp", "model", "feature_dim", "loss_type", "val_acc", "macro_f1", "runtime"]

    def __init__(self, path: str = "output/leaderboard"):
        """
        Args:
            path: 输出文件前缀（不含扩展名）
        """
        self.path = path
        self.entries: list[dict] = []
        self._load()

    def add(
        self,
        model_name: str,
        feature_dim: int = 767,
        loss_type: str = "ce",
        val_acc: float = 0.0,
        macro_f1: float = 0.0,
        runtime: float = 0.0,
        config: dict | None = None,
    ):
        """添加一条排行榜记录。"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "feature_dim": feature_dim,
            "loss_type": loss_type,
            "val_acc": round(val_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "runtime": round(runtime, 2),
        }
        if config:
            entry["config"] = config

        self.entries.append(entry)
        self._sort()
        self._save_json()
        self._save_csv()

    def _sort(self):
        """按 val_acc 降序排列，保留 Top-10。"""
        self.entries.sort(key=lambda e: e["val_acc"], reverse=True)
        self.entries = self.entries[:10]

    def top_k(self, k: int = 3) -> list[dict]:
        """返回 Top-K 结果。"""
        return self.entries[:k]

    def best(self) -> dict | None:
        """返回最佳结果。"""
        return self.entries[0] if self.entries else None

    def display(self):
        """显示排行榜。"""
        if not self.entries:
            print("[LEADERBOARD] 暂无记录")
            return

        print("\n" + "=" * 48)
        for rank, entry in enumerate(self.entries[:5], 1):
            loss = entry.get("loss_type", "ce")
            model = entry["model"]
            if loss != "ce":
                model = f"{model} + FocalLoss"
            print(f"Rank{rank}")
            print(f"  {model}")
            print(f"  Acc={entry['val_acc']:.4f}")
            f1 = entry.get("macro_f1", 0)
            if f1:
                print(f"  MacroF1={f1:.4f}")
            print("-" * 48)
        print()

    def _save_json(self):
        """保存 JSON。"""
        json_path = self.path + ".json"
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def _save_csv(self):
        """保存 CSV。"""
        csv_path = self.path + ".csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for entry in self.entries:
                row = {col: entry.get(col, "") for col in self.CSV_COLUMNS}
                writer.writerow(row)

    def _load(self):
        """从 JSON 文件加载。"""
        json_path = self.path + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

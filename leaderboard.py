"""排行榜输出：实时记录最佳模型表现。"""
import json
import os
from datetime import datetime


class Leaderboard:
    """实时排行榜，记录最佳模型表现。

    自动保存 Top-10 实验结果。
    """

    def __init__(self, path: str = "output/leaderboard.json"):
        self.path = path
        self.entries: list[dict] = []
        self._load()

    def add(self, model_name: str, feature: str, val_acc: float, runtime: float = 0.0, config: dict | None = None):
        """添加一条排行榜记录。"""
        entry = {
            "model": model_name,
            "feature": feature,
            "val_acc": round(val_acc, 4),
            "runtime": round(runtime, 2),
            "timestamp": datetime.now().isoformat(),
        }
        if config:
            entry["config"] = config

        self.entries.append(entry)
        self._sort()
        self._save()

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
            print(f"Rank{rank}")
            print(f"  {entry['model']}")
            print(f"  Feature={entry['feature']}")
            print(f"  Acc={entry['val_acc']:.4f}")
            if entry.get("runtime"):
                print(f"  Runtime={entry['runtime']:.1f}s")
            print("-" * 48)
        print()

    def _save(self):
        """保存到文件。"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

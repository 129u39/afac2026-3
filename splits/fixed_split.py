"""固定验证集划分：所有模型共用。"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

CACHE_DIR = "cache"
CACHE_PATH = os.path.join(CACHE_DIR, "split.pkl")


def get_fixed_split(
    train_idx: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """获取固定的 train/val 划分（首次生成后缓存）。

    Args:
        train_idx: 全部训练节点索引
        labels: 完整标签
        val_ratio: 验证比例
        seed: 随机种子

    Returns:
        (train_sub, val_sub)
    """
    path = CACHE_PATH

    # 尝试从缓存加载
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            print(f"[SPLIT] 加载缓存: train={len(cached['train'])} val={len(cached['val'])}")
            return cached["train"], cached["val"]
        except Exception:
            pass

    # 首次生成
    train_sub, val_sub = train_test_split(
        train_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels[train_idx],
    )

    # 保存缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"train": train_sub, "val": val_sub}, f)

    print(f"[SPLIT] 新划分: train={len(train_sub)} val={len(val_sub)} seed={seed}")
    return train_sub, val_sub

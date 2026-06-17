"""固定验证集划分器：所有模型共用同一组 train/val 划分。

解决 train_gnn() 切一次、evaluate() 再切一次导致的"训练目标 != 搜索目标"问题。
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def get_split(
    train_idx: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
    cache_path: str = "output/split_cache.pkl",
) -> tuple[np.ndarray, np.ndarray]:
    """获取固定的 train/val 划分。

    首次调用时生成并缓存，后续直接加载缓存。

    Args:
        train_idx: 全部训练节点索引
        labels: 全部标签 (完整长度)
        val_ratio: 验证比例
        seed: 随机种子
        cache_path: 缓存文件路径

    Returns:
        (train_sub, val_sub): 训练子集和验证子集索引
    """
    # 尝试从缓存加载
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                print(f"[SPLIT] 加载缓存: train={len(cached['train'])} val={len(cached['val'])}")
                return cached["train"], cached["val"]
        except Exception:
            pass

    # 生成划分
    train_sub, val_sub = train_test_split(
        train_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels[train_idx],
    )

    # 保存缓存
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"train": train_sub, "val": val_sub}, f)

    print(f"[SPLIT] 新划分: train={len(train_sub)} val={len(val_sub)} seed={seed}")
    return train_sub, val_sub

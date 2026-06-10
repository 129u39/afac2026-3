"""特征缓存：避免重复计算。"""

import os
import torch
import numpy as np


class FeatureCache:
    """特征缓存管理器。

    缓存预计算的特征，避免重复计算。
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save(self, name: str, features: np.ndarray):
        """保存特征到缓存。"""
        path = os.path.join(self.cache_dir, f"{name}.pt")
        torch.save(torch.tensor(features, dtype=torch.float32), path)

    def load(self, name: str) -> np.ndarray | None:
        """从缓存加载特征。"""
        path = os.path.join(self.cache_dir, f"{name}.pt")
        if os.path.exists(path):
            return torch.load(path).numpy()
        return None

    def exists(self, name: str) -> bool:
        """检查缓存是否存在。"""
        path = os.path.join(self.cache_dir, f"{name}.pt")
        return os.path.exists(path)

"""模型工具函数。"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置全局随机种子以保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取计算设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

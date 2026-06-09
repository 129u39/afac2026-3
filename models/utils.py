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


def enable_amp():
    """启用自动混合精度训练。

    在 GPU 上使用 float16 加速，在 CPU 上使用 bfloat16。
    返回 GradScaler 实例。
    """
    if torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def get_optimal_batch_size(model: nn.Module, device: torch.device) -> int:
    """根据 GPU 显存自动选择最优 batch size。"""
    if not torch.cuda.is_available():
        return 256

    # 获取 GPU 显存
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    # 根据显存大小选择 batch size
    if gpu_memory >= 16:  # V100/A100
        return 1024
    elif gpu_memory >= 8:  # RTX 3070/4070
        return 512
    elif gpu_memory >= 4:  # RTX 3060/4060
        return 256
    else:
        return 128


def enable_tf32():
    """启用 TF32 加速（Ampere+ GPU）。"""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

"""类别平衡权重：N / (num_classes * count[c])。

替换 F.cross_entropy() 为 nn.CrossEntropyLoss(weight=class_weights)。
"""
import numpy as np
import torch


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """计算类别平衡权重。

    weights[c] = N / (num_classes * count[c])

    Args:
        labels: 训练标签数组

    Returns:
        权重张量, shape (num_classes,)
    """
    classes = np.unique(labels)
    num_classes = len(classes)
    N = len(labels)
    counts = np.bincount(labels.astype(int), minlength=num_classes)

    weights = N / (num_classes * counts.astype(np.float64))
    weights = weights / weights.sum() * num_classes  # 归一化

    w_min = weights.min()
    w_max = weights.max()
    print(f"[CLASS_WEIGHT] min={w_min:.2f} max={w_max:.2f}")

    return torch.tensor(weights, dtype=torch.float32)

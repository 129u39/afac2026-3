"""加权交叉熵损失：处理类别不平衡。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失。

    使用逆频率加权处理类别不平衡。
    weights = 1 / log(freq + 1)
    """

    def __init__(self, class_counts: np.ndarray):
        """
        Args:
            class_counts: 每个类别的样本数
        """
        super().__init__()
        # 计算权重: 1 / log(freq + 1)
        weights = 1.0 / np.log(class_counts + 1)
        weights = weights / weights.sum() * len(weights)  # 归一化
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) 预测 logits
            targets: (batch_size,) 真实标签

        返回:
            loss: 标量
        """
        return F.cross_entropy(inputs, targets, weight=self.weights)

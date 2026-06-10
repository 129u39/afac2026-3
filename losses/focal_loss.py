"""Focal Loss：处理类别不平衡。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss。

    用于处理类别不平衡问题。
    通过降低易分类样本的权重，聚焦于难分类样本。

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        """
        Args:
            gamma: 聚焦参数，越大越关注难样本
            alpha: 类别权重，shape (num_classes,)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) 预测 logits
            targets: (batch_size,) 真实标签

        返回:
            loss: 标量
        """
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # 计算 p_t
        p_t = (probs * targets_one_hot).sum(dim=1)

        # 计算 focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 应用 focal weight
        loss = focal_weight * ce_loss

        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()

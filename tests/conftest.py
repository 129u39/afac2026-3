"""Pytest 配置和共享 fixtures。"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_cls_config():
    """示例分类配置。"""
    return {
        "model_type": "GraphSAGE",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "epochs": 200,
        "patience": 30,
    }


@pytest.fixture
def sample_rec_config():
    """示例推荐配置。"""
    return {
        "model_type": "BPR_MF",
        "embedding_dim": 64,
        "lr": 0.01,
        "epochs": 50,
        "batch_size": 512,
        "weight_decay": 1e-5,
    }


@pytest.fixture
def sample_memory():
    """示例实验记忆。"""
    from memory import ExperimentMemory, ExperimentRecord

    mem = ExperimentMemory()
    mem.add(ExperimentRecord(0, "GCN", {"hidden_dim": 64}, {"val_accuracy": 0.7}, 10))
    mem.add(ExperimentRecord(1, "GAT", {"hidden_dim": 128}, {"val_accuracy": 0.75}, 15))
    mem.add(ExperimentRecord(2, "GraphSAGE", {"hidden_dim": 256}, {"val_accuracy": 0.8}, 20))
    return mem

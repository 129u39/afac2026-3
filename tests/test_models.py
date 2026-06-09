"""测试模型模块。"""

import pytest
import torch
import numpy as np

from models.utils import set_seed, get_device


class TestUtils:
    """测试模型工具函数。"""

    def test_set_seed(self):
        """测试设置随机种子。"""
        set_seed(42)
        a = torch.rand(3)
        set_seed(42)
        b = torch.rand(3)
        assert torch.allclose(a, b)

    def test_get_device(self):
        """测试获取设备。"""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]


class TestGNNClassifier:
    """测试 GNN 分类器。"""

    def test_create_gcn(self):
        """测试创建 GCN 模型。"""
        from models.gnn_classifier import GNNClassifier
        model = GNNClassifier(in_dim=64, hidden_dim=32, num_classes=10, model_type="GCN")
        assert model is not None

    def test_create_gat(self):
        """测试创建 GAT 模型。"""
        from models.gnn_classifier import GNNClassifier
        model = GNNClassifier(in_dim=64, hidden_dim=32, num_classes=10, model_type="GAT")
        assert model is not None

    def test_create_graphsage(self):
        """测试创建 GraphSAGE 模型。"""
        from models.gnn_classifier import GNNClassifier
        model = GNNClassifier(in_dim=64, hidden_dim=32, num_classes=10, model_type="GraphSAGE")
        assert model is not None

    def test_invalid_model_type(self):
        """测试无效模型类型。"""
        from models.gnn_classifier import GNNClassifier
        with pytest.raises(ValueError):
            GNNClassifier(in_dim=64, hidden_dim=32, num_classes=10, model_type="Invalid")


class TestAPPNP:
    """测试 APPNP 模型。"""

    def test_create(self):
        """测试创建 APPNP 模型。"""
        from models.appnp import APPNP
        model = APPNP(in_dim=64, hidden_dim=32, num_classes=10)
        assert model is not None


class TestGCNII:
    """测试 GCNII 模型。"""

    def test_create(self):
        """测试创建 GCNII 模型。"""
        from models.gcnii import GCNII
        model = GCNII(in_dim=64, hidden_dim=32, num_classes=10, num_layers=4)
        assert model is not None


class TestMLPBaseline:
    """测试 MLP 基线模型。"""

    def test_create(self):
        """测试创建 MLP 模型。"""
        from models.mlp_baseline import MLPBaseline
        model = MLPBaseline(in_dim=64, hidden_dim=32, num_classes=10)
        assert model is not None


class TestRecommender:
    """测试推荐模型。"""

    def test_create_popularity(self):
        """测试创建 Popularity 推荐器。"""
        from models.recommender import RecommenderSystem
        rec = RecommenderSystem(model_type="Popularity")
        assert rec.model_type == "Popularity"

    def test_create_bpr(self):
        """测试创建 BPR 推荐器。"""
        from models.recommender import RecommenderSystem
        rec = RecommenderSystem(model_type="BPR_MF", embedding_dim=32)
        assert rec.model_type == "BPR_MF"

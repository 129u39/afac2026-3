"""集成测试：验证模型训练是否正常工作。"""

import pytest
import torch
import numpy as np
import time

from models.utils import set_seed, get_device
from data_loader import load_classification, classification_to_pyg, load_recommendation


@pytest.fixture(scope="module")
def cls_data():
    """加载分类数据。"""
    set_seed(42)
    return load_classification("data/A分类/A分类/A1.npz")


@pytest.fixture(scope="module")
def rec_data():
    """加载推荐数据。"""
    set_seed(42)
    return load_recommendation("data/A推荐/A推荐")


@pytest.fixture(scope="module")
def device():
    """获取设备。"""
    return get_device()


class TestClassificationTraining:
    """测试分类模型训练。"""

    def test_gcn_training(self, cls_data, device):
        """测试 GCN 训练。"""
        from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = GNNClassifier(
            in_dim=cls_data["num_features"],
            hidden_dim=64,
            num_classes=cls_data["num_classes"],
            num_layers=2,
            model_type="GCN",
            dropout=0.5,
        ).to(device)

        result = train_gnn(model, pyg_data, lr=0.01, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"GCN accuracy too low: {result['best_val_acc']}"
        assert len(result["train_losses"]) > 0
        print(f"GCN: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_graphsage_training(self, cls_data, device):
        """测试 GraphSAGE 训练。"""
        from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = GNNClassifier(
            in_dim=cls_data["num_features"],
            hidden_dim=128,
            num_classes=cls_data["num_classes"],
            num_layers=3,
            model_type="GraphSAGE",
            dropout=0.1,
        ).to(device)

        result = train_gnn(model, pyg_data, lr=0.005, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"GraphSAGE accuracy too low: {result['best_val_acc']}"
        print(f"GraphSAGE: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_gat_training(self, cls_data, device):
        """测试 GAT 训练。"""
        from models.gnn_classifier import GNNClassifier, train_gnn, predict_gnn

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = GNNClassifier(
            in_dim=cls_data["num_features"],
            hidden_dim=64,
            num_classes=cls_data["num_classes"],
            num_layers=2,
            model_type="GAT",
            dropout=0.5,
        ).to(device)

        result = train_gnn(model, pyg_data, lr=0.005, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"GAT accuracy too low: {result['best_val_acc']}"
        print(f"GAT: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_appnp_training(self, cls_data, device):
        """测试 APPNP 训练。"""
        from models.appnp import APPNP, train_appnp

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = APPNP(
            in_dim=cls_data["num_features"],
            hidden_dim=128,
            num_classes=cls_data["num_classes"],
            num_layers=2,
            dropout=0.3,
            K=10,
            alpha=0.1,
        ).to(device)

        result = train_appnp(model, pyg_data, lr=0.005, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"APPNP accuracy too low: {result['best_val_acc']}"
        print(f"APPNP: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_gcnii_training(self, cls_data, device):
        """测试 GCNII 训练。"""
        from models.gcnii import GCNII, train_gcnii

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = GCNII(
            in_dim=cls_data["num_features"],
            hidden_dim=64,
            num_classes=cls_data["num_classes"],
            num_layers=4,
            dropout=0.5,
            alpha=0.1,
            theta=0.5,
        ).to(device)

        result = train_gcnii(model, pyg_data, lr=0.01, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"GCNII accuracy too low: {result['best_val_acc']}"
        print(f"GCNII: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_mlp_training(self, cls_data, device):
        """测试 MLP 训练（基线）。"""
        from models.mlp_baseline import MLPBaseline, train_mlp

        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)

        model = MLPBaseline(
            in_dim=cls_data["num_features"],
            hidden_dim=128,
            num_classes=cls_data["num_classes"],
            num_layers=2,
            dropout=0.5,
        ).to(device)

        result = train_mlp(model, pyg_data, lr=0.01, epochs=20, patience=10)

        assert result["best_val_acc"] > 0.1, f"MLP accuracy too low: {result['best_val_acc']}"
        print(f"MLP: val_acc={result['best_val_acc']:.4f}, epochs={len(result['train_losses'])}")

    def test_model_comparison(self, cls_data, device):
        """比较不同模型的表现。"""
        from models.gnn_classifier import GNNClassifier, train_gnn
        from models.appnp import APPNP, train_appnp
        from models.mlp_baseline import MLPBaseline, train_mlp

        results = {}
        epochs = 15

        # GCN
        set_seed(42)
        pyg_data = classification_to_pyg(cls_data, device)
        model = GNNClassifier(cls_data["num_features"], 64, cls_data["num_classes"], 2, "GCN", 0.5).to(device)
        r = train_gnn(model, pyg_data, lr=0.01, epochs=epochs, patience=10)
        results["GCN"] = r["best_val_acc"]

        # GraphSAGE
        set_seed(42)
        model = GNNClassifier(cls_data["num_features"], 128, cls_data["num_classes"], 3, "GraphSAGE", 0.1).to(device)
        r = train_gnn(model, pyg_data, lr=0.005, epochs=epochs, patience=10)
        results["GraphSAGE"] = r["best_val_acc"]

        # APPNP
        set_seed(42)
        model = APPNP(cls_data["num_features"], 128, cls_data["num_classes"], 2, 0.3, 10, 0.1).to(device)
        r = train_appnp(model, pyg_data, lr=0.005, epochs=epochs, patience=10)
        results["APPNP"] = r["best_val_acc"]

        # MLP
        set_seed(42)
        model = MLPBaseline(cls_data["num_features"], 128, cls_data["num_classes"], 2, 0.5).to(device)
        r = train_mlp(model, pyg_data, lr=0.01, epochs=epochs, patience=10)
        results["MLP"] = r["best_val_acc"]

        print("\n=== 模型比较 ===")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {acc:.4f}")

        # 验证 GNN 优于 MLP（图结构有贡献）
        best_gnn = max(results["GCN"], results["GraphSAGE"], results["APPNP"])
        assert best_gnn >= results["MLP"] * 0.9, "GNN should be competitive with MLP"


class TestRecommendationTraining:
    """测试推荐模型训练。"""

    def test_popularity(self, rec_data):
        """测试 Popularity 推荐。"""
        from models.recommender import RecommenderSystem

        rec = RecommenderSystem(model_type="Popularity")
        rec.fit(rec_data)

        # 测试预测
        test_df = rec_data["test_df"]
        uid = test_df.iloc[0]["uid"]
        from data_loader import parse_seq_dedup
        seq = parse_seq_dedup(test_df.iloc[0]["item_seq_dedup"])

        preds = rec.predict(uid, seq, top_k=10)
        assert len(preds) == 10
        print(f"Popularity: predicted {len(preds)} items")

    def test_itemcf(self, rec_data):
        """测试 ItemCF 推荐。"""
        from models.recommender import RecommenderSystem

        rec = RecommenderSystem(model_type="ItemCF")
        start = time.time()
        rec.fit(rec_data)
        train_time = time.time() - start

        # 测试预测
        test_df = rec_data["test_df"]
        uid = test_df.iloc[0]["uid"]
        from data_loader import parse_seq_dedup
        seq = parse_seq_dedup(test_df.iloc[0]["item_seq_dedup"])

        preds = rec.predict(uid, seq, top_k=10)
        assert len(preds) == 10
        print(f"ItemCF: predicted {len(preds)} items, train_time={train_time:.1f}s")

    def test_bpr(self, rec_data):
        """测试 BPR-MF 推荐。"""
        from models.recommender import RecommenderSystem

        rec = RecommenderSystem(model_type="BPR_MF", embedding_dim=32, epochs=10, batch_size=256)
        start = time.time()
        rec.fit(rec_data)
        train_time = time.time() - start

        # 测试预测
        test_df = rec_data["test_df"]
        uid = test_df.iloc[0]["uid"]
        from data_loader import parse_seq_dedup
        seq = parse_seq_dedup(test_df.iloc[0]["item_seq_dedup"])

        preds = rec.predict(uid, seq, top_k=10)
        assert len(preds) == 10
        print(f"BPR-MF: predicted {len(preds)} items, train_time={train_time:.1f}s")


class TestRunnerIntegration:
    """测试实验运行器集成。"""

    def test_runner_classification(self, cls_data, device):
        """测试运行器分类任务。"""
        from runner.experiment_runner import ExperimentRunner

        runner = ExperimentRunner("classification", cls_data, device)
        config = {
            "model_type": "GCN",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "epochs": 10,
            "patience": 5,
        }

        result = runner.run(config)

        assert result.status == "success"
        assert result.metric > 0.1
        assert result.train_time > 0
        print(f"Runner GCN: metric={result.metric:.4f}, time={result.train_time:.1f}s")

    def test_runner_recommendation(self, rec_data, device):
        """测试运行器推荐任务。"""
        from runner.experiment_runner import ExperimentRunner

        runner = ExperimentRunner("recommendation", rec_data, device)
        config = {
            "model_type": "Popularity",
        }

        result = runner.run(config)

        assert result.status == "success"
        assert result.metric >= 0
        print(f"Runner Popularity: metric={result.metric:.4f}, time={result.train_time:.1f}s")

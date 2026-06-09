"""测试实验记忆模块。"""

import json
import os
import pytest
import tempfile

from memory import ExperimentMemory, ExperimentRecord


class TestExperimentRecord:
    """测试 ExperimentRecord。"""

    def test_create_record(self):
        """测试创建记录。"""
        rec = ExperimentRecord(
            round_num=0,
            model_type="GCN",
            config={"hidden_dim": 64},
            metrics={"val_accuracy": 0.8},
            elapsed_seconds=10.0,
        )
        assert rec.round_num == 0
        assert rec.model_type == "GCN"
        assert rec.metrics["val_accuracy"] == 0.8
        assert len(rec.exp_id) == 8
        assert rec.timestamp != ""

    def test_auto_fields(self):
        """测试自动生成的字段。"""
        rec = ExperimentRecord(
            round_num=0,
            model_type="GCN",
            config={},
            metrics={},
            elapsed_seconds=0,
        )
        assert rec.exp_id != ""
        assert rec.timestamp != ""
        assert rec.status == "success"


class TestExperimentMemory:
    """测试 ExperimentMemory。"""

    def test_add_record(self):
        """测试添加记录。"""
        mem = ExperimentMemory()
        rec = ExperimentRecord(
            round_num=0,
            model_type="GCN",
            config={"hidden_dim": 64},
            metrics={"val_accuracy": 0.8},
            elapsed_seconds=10.0,
        )
        mem.add(rec)
        assert len(mem.records) == 1

    def test_get_best_classification(self):
        """测试获取最佳分类记录。"""
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.7}, 10))
        mem.add(ExperimentRecord(1, "GCN", {}, {"val_accuracy": 0.8}, 10))
        mem.add(ExperimentRecord(2, "GCN", {}, {"val_accuracy": 0.75}, 10))

        best = mem.get_best("classification")
        assert best.metrics["val_accuracy"] == 0.8
        assert best.round_num == 1

    def test_get_best_recommendation(self):
        """测试获取最佳推荐记录。"""
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "BPR", {}, {"ndcg@k": 0.1}, 10))
        mem.add(ExperimentRecord(1, "BPR", {}, {"ndcg@k": 0.15}, 10))

        best = mem.get_best("recommendation")
        assert best.metrics["ndcg@k"] == 0.15

    def test_get_last_k(self):
        """测试获取最近 k 条记录。"""
        mem = ExperimentMemory()
        for i in range(10):
            mem.add(ExperimentRecord(i, "GCN", {}, {"val_accuracy": 0.5 + i * 0.01}, 10))

        last_3 = mem.get_last_k(3)
        assert len(last_3) == 3
        assert last_3[0].round_num == 7
        assert last_3[2].round_num == 9

    def test_get_by_model(self):
        """测试按模型类型筛选。"""
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.7}, 10))
        mem.add(ExperimentRecord(1, "GAT", {}, {"val_accuracy": 0.75}, 10))
        mem.add(ExperimentRecord(2, "GCN", {}, {"val_accuracy": 0.8}, 10))

        gcn_records = mem.get_by_model("GCN")
        assert len(gcn_records) == 2

    def test_save_load(self):
        """测试保存和加载。"""
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {"hidden_dim": 64}, {"val_accuracy": 0.8}, 10))
        mem.add(ExperimentRecord(1, "GAT", {"hidden_dim": 128}, {"val_accuracy": 0.75}, 10))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem.save(path)

            mem2 = ExperimentMemory()
            mem2.load(path)

            assert len(mem2.records) == 2
            assert mem2.get_best("classification").metrics["val_accuracy"] == 0.8
        finally:
            os.unlink(path)

    def test_summary(self):
        """测试摘要生成。"""
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.8}, 10))

        summary = mem.summary()
        assert "1" in summary
        assert "GCN" in summary

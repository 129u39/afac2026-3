"""测试 TopK Pool。"""

import json
import os
import pytest
import tempfile

from search.topk_pool import TopKPool


class TestTopKPool:
    """测试 TopKPool。"""

    def test_init(self):
        """测试初始化。"""
        pool = TopKPool(max_size=5)
        assert pool.max_size == 5
        assert len(pool) == 0

    def test_add(self):
        """测试添加配置。"""
        pool = TopKPool(max_size=5)
        pool.add({"model_type": "GCN"}, 0.8)
        assert len(pool) == 1

    def test_add_multiple(self):
        """测试添加多个配置。"""
        pool = TopKPool(max_size=5)
        pool.add({"model_type": "GCN"}, 0.7)
        pool.add({"model_type": "GAT"}, 0.8)
        pool.add({"model_type": "GraphSAGE"}, 0.75)
        assert len(pool) == 3

    def test_max_size(self):
        """测试最大容量限制。"""
        pool = TopKPool(max_size=3)
        for i in range(5):
            pool.add({"model_type": f"Model{i}"}, 0.5 + i * 0.1)
        assert len(pool) == 3

    def test_get_top_k(self):
        """测试获取 Top-K。"""
        pool = TopKPool(max_size=10)
        pool.add({"model_type": "GCN"}, 0.7)
        pool.add({"model_type": "GAT"}, 0.8)
        pool.add({"model_type": "GraphSAGE"}, 0.75)

        top2 = pool.get_top_k(2)
        assert len(top2) == 2
        assert top2[0]["model_type"] == "GAT"
        assert top2[1]["model_type"] == "GraphSAGE"

    def test_get_best(self):
        """测试获取最佳配置。"""
        pool = TopKPool()
        pool.add({"model_type": "GCN"}, 0.7)
        pool.add({"model_type": "GAT"}, 0.8)

        best = pool.get_best()
        assert best["model_type"] == "GAT"

    def test_get_best_empty(self):
        """测试空池获取最佳配置。"""
        pool = TopKPool()
        assert pool.get_best() is None

    def test_get_best_metric(self):
        """测试获取最佳指标。"""
        pool = TopKPool()
        pool.add({"model_type": "GCN"}, 0.7)
        pool.add({"model_type": "GAT"}, 0.8)

        assert pool.get_best_metric() == 0.8

    def test_update_existing(self):
        """测试更新已有配置。"""
        pool = TopKPool()
        config = {"model_type": "GCN", "hidden_dim": 64}
        pool.add(config, 0.7)
        pool.add(config, 0.8)  # 更新

        assert len(pool) == 1
        assert pool.get_best_metric() == 0.8

    def test_get_focus_configs(self):
        """测试获取聚焦配置。"""
        pool = TopKPool()
        pool.add({"model_type": "GCN"}, 0.8)
        pool.add({"model_type": "GAT"}, 0.7)

        configs = pool.get_focus_configs(n=10, focus_ratio=0.8)
        assert len(configs) == 10

    def test_save_load(self):
        """测试保存和加载。"""
        pool = TopKPool(max_size=5)
        pool.add({"model_type": "GCN"}, 0.8)
        pool.add({"model_type": "GAT"}, 0.7)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            pool.path = path
            pool.save()

            pool2 = TopKPool(path=path)
            assert len(pool2) == 2
            assert pool2.get_best_metric() == 0.8
        finally:
            os.unlink(path)

    def test_summary(self):
        """测试摘要生成。"""
        pool = TopKPool()
        pool.add({"model_type": "GCN"}, 0.8)

        summary = pool.summary()
        assert "1" in summary

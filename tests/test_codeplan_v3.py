"""CODE_PLAN v3.0 新增组件测试。"""
import numpy as np
import pytest


class TestDataReport:
    """Phase 0: 数据报告测试。"""

    def test_generate_report(self):
        from analysis.data_report import generate_report
        from scipy.sparse import csr_matrix

        data = {
            "adj_csr": csr_matrix((5, 5)),
            "features": csr_matrix(np.eye(5)),
            "labels": np.array([0, 1, 0, 1, -1], dtype=np.int64),
            "train_idx": np.array([0, 1, 2, 3]),
            "test_idx": np.array([4]),
        }
        report = generate_report(data, logger=lambda _: None)
        assert report["nodes"] == 5
        assert report["features"] == 5
        assert report["classes"] == 2
        assert "class_distribution" in report
        assert "imbalance_ratio" in report
        assert "isolated_ratio" in report
        print(f"[TEST] data_report OK: nodes={report['nodes']}")


class TestFixedSplit:
    """Phase 1: 固定验证集测试。"""

    def test_get_fixed_split(self):
        import tempfile, os, pickle
        from splits.fixed_split import get_fixed_split

        train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "split.pkl")
            train_sub, val_sub = get_fixed_split(
                train_idx, labels, val_ratio=0.2, seed=42,
            )
            assert len(train_sub) == 8
            assert len(val_sub) == 2
            # 第二次调用走缓存
            train_sub2, val_sub2 = get_fixed_split(
                train_idx, labels, val_ratio=0.2, seed=42,
            )
            assert np.array_equal(train_sub, train_sub2)
            assert np.array_equal(val_sub, val_sub2)
            print(f"[TEST] fixed_split OK: train={len(train_sub)} val={len(val_sub)}")


class TestClassBalanced:
    """Phase 2: 类别平衡权重测试。"""

    def test_compute_class_weights(self):
        from losses.class_balanced import compute_class_weights
        labels = np.array([0, 0, 0, 1, 1, 2])
        w = compute_class_weights(labels)
        assert len(w) == 3
        assert w[0] < w[1] < w[2]  # 样本越少权重越大
        print(f"[TEST] class_balanced OK: weights={w.numpy().round(3)}")


class TestNodeStats:
    """Phase 6: 节点统计特征测试。"""

    def test_compute_node_stats(self):
        from features.node_stats import compute_node_stats
        from scipy.sparse import csr_matrix

        # 3个节点，0-1相连，2孤立
        adj = csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(3, 3))
        feat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        combined = compute_node_stats(adj, feat)
        # feat(2) + degree(1) + is_isolated(1) = 4
        assert combined.shape == (3, 4)
        assert combined[2, 2] == 0.0  # degree=0
        assert combined[2, 3] == 1.0  # is_isolated=1
        print(f"[TEST] node_stats OK: shape={combined.shape}")


class TestLeaderboard:
    """Phase 10: 排行榜测试。"""

    def test_leaderboard(self):
        import tempfile, os
        from leaderboard import Leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lb")
            lb = Leaderboard(path)
            lb.add("GraphSAGE", feature_dim=128, loss_type="focal", val_acc=0.78, macro_f1=0.45)
            lb.add("GCNII", feature_dim=256, loss_type="ce", val_acc=0.76, macro_f1=0.42)
            best = lb.best()
            assert best["model"] == "GraphSAGE"
            assert best["val_acc"] == 0.78
            # 验证 CSV 输出
            csv_path = path + ".csv"
            assert os.path.exists(csv_path)
            print(f"[TEST] leaderboard OK: top={best['model']} acc={best['val_acc']}")

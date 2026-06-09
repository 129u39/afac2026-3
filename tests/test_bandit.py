"""测试 UCB Bandit。"""

import pytest

from planner.bandit import UCBBandit


class TestUCBBandit:
    """测试 UCBBandit。"""

    def test_init(self):
        """测试初始化。"""
        bandit = UCBBandit(arms=["GCN", "GAT", "GraphSAGE"])
        assert len(bandit.arms) == 3
        assert bandit.total_pulls == 0

    def test_select_unexplored(self):
        """测试优先选择未探索的臂。"""
        bandit = UCBBandit(arms=["GCN", "GAT", "GraphSAGE"])
        arm = bandit.select_arm()
        assert arm in ["GCN", "GAT", "GraphSAGE"]

    def test_update(self):
        """测试更新。"""
        bandit = UCBBandit(arms=["GCN", "GAT", "GraphSAGE"])
        bandit.update("GCN", 0.8)
        assert bandit.arms["GCN"].pulls == 1
        assert bandit.arms["GCN"].total_reward == 0.8
        assert bandit.total_pulls == 1

    def test_update_invalid_arm(self):
        """测试更新无效臂。"""
        bandit = UCBBandit(arms=["GCN", "GAT"])
        with pytest.raises(ValueError):
            bandit.update("Invalid", 0.8)

    def test_explore_all_arms(self):
        """测试探索所有臂。"""
        bandit = UCBBandit(arms=["GCN", "GAT", "GraphSAGE"])

        # 第一轮：选择未探索的臂
        arm1 = bandit.select_arm()
        bandit.update(arm1, 0.7)

        # 第二轮：选择另一个未探索的臂
        arm2 = bandit.select_arm()
        bandit.update(arm2, 0.8)

        # 第三轮：选择最后一个未探索的臂
        arm3 = bandit.select_arm()
        bandit.update(arm3, 0.9)

        # 验证所有臂都被探索过
        assert bandit.total_pulls == 3
        for name, stats in bandit.arms.items():
            assert stats.pulls == 1

    def test_exploit_best_arm(self):
        """测试利用最佳臂。"""
        bandit = UCBBandit(arms=["GCN", "GAT"], c=0.1)  # 低探索系数

        # 给 GCN 高奖励
        bandit.update("GCN", 0.9)
        bandit.update("GAT", 0.1)

        # 应该倾向于选择 GCN
        gc = 0
        for _ in range(100):
            arm = bandit.select_arm()
            if arm == "GCN":
                gc += 1
            bandit.update(arm, 0.5)

        # GCN 应该被选择更多次
        assert gc > 50

    def test_compute_aware_update(self):
        """测试 Compute-Aware 更新。"""
        bandit = UCBBandit(arms=["GCN", "GAT"])

        # 快速提升
        bandit.update_compute_aware("GCN", 0.8, 0.7, 10.0)
        # 慢速提升
        bandit.update_compute_aware("GAT", 0.8, 0.7, 100.0)

        # GCN 的 reward 应该更高
        assert bandit.arms["GCN"].total_reward > bandit.arms["GAT"].total_reward

    def test_get_stats(self):
        """测试获取统计信息。"""
        bandit = UCBBandit(arms=["GCN", "GAT"])
        bandit.update("GCN", 0.8)

        stats = bandit.get_stats()
        assert "GCN" in stats
        assert stats["GCN"]["pulls"] == 1
        assert stats["GCN"]["mean_reward"] == 0.8

    def test_get_arm_names(self):
        """测试获取臂名称。"""
        bandit = UCBBandit(arms=["GCN", "GAT", "GraphSAGE"])
        names = bandit.get_arm_names()
        assert set(names) == {"GCN", "GAT", "GraphSAGE"}

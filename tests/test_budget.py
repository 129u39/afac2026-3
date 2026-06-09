"""测试预算管理器。"""

import pytest
import time

from budget_manager import BudgetManager


class TestBudgetManager:
    """测试 BudgetManager。"""

    def test_init(self):
        """测试初始化。"""
        bm = BudgetManager(total_seconds=7200, safety_margin=300)
        assert bm.total_seconds == 7200
        assert bm.safety_margin == 300
        assert bm.effective_budget == 6900

    def test_elapsed(self):
        """测试已用时间。"""
        bm = BudgetManager()
        time.sleep(0.1)
        assert bm.elapsed() >= 0.1

    def test_remaining(self):
        """测试剩余时间。"""
        bm = BudgetManager(total_seconds=100, safety_margin=10)
        assert bm.remaining() <= 90

    def test_should_continue_initial(self):
        """测试初始状态应继续。"""
        bm = BudgetManager(total_seconds=7200, safety_margin=300)
        assert bm.should_continue() is True

    def test_should_continue_no_improve(self):
        """测试连续无提升应停止。"""
        bm = BudgetManager(total_seconds=7200, safety_margin=300, max_no_improve_rounds=3)
        for _ in range(3):
            bm.record_improvement(False)
        assert bm.should_continue() is False

    def test_should_continue_with_improvement(self):
        """测试有提升应继续。"""
        bm = BudgetManager(total_seconds=7200, safety_margin=300, max_no_improve_rounds=3)
        bm.record_improvement(False)
        bm.record_improvement(False)
        bm.record_improvement(True)  # 有提升
        assert bm.should_continue() is True

    def test_record_round(self):
        """测试记录轮次。"""
        bm = BudgetManager()
        bm.record_round(10.0)
        bm.record_round(20.0)
        assert len(bm.round_times) == 2
        assert bm.avg_round_time() == 15.0

    def test_can_run(self):
        """测试能否运行。"""
        bm = BudgetManager(total_seconds=7200, safety_margin=300)
        assert bm.can_run({"model_type": "GCN"}) is True

    def test_remaining_budget(self):
        """测试剩余预算信息。"""
        bm = BudgetManager()
        budget = bm.remaining_budget()
        assert "elapsed_seconds" in budget
        assert "remaining_seconds" in budget
        assert "should_continue" in budget

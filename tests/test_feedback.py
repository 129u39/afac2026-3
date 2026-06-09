"""测试反馈分析器。"""

import pytest

from memory import ExperimentMemory, ExperimentRecord
from feedback_analyzer import FeedbackAnalyzer


class TestFeedbackAnalyzer:
    """测试 FeedbackAnalyzer。"""

    def test_analyze_empty(self):
        """测试分析空记忆。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        result = analyzer.analyze(mem, "classification")
        assert result["trend"] == "no_data"

    def test_analyze_improving(self):
        """测试分析改善趋势。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        # 添加更多数据点以减少方差
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.7}, 10))
        mem.add(ExperimentRecord(1, "GCN", {}, {"val_accuracy": 0.72}, 10))
        mem.add(ExperimentRecord(2, "GCN", {}, {"val_accuracy": 0.74}, 10))
        mem.add(ExperimentRecord(3, "GCN", {}, {"val_accuracy": 0.76}, 10))
        mem.add(ExperimentRecord(4, "GCN", {}, {"val_accuracy": 0.78}, 10))
        mem.add(ExperimentRecord(5, "GCN", {}, {"val_accuracy": 0.8}, 10))

        result = analyzer.analyze(mem, "classification")
        assert result["trend"] in ["improving", "unstable"]

    def test_analyze_plateau(self):
        """测试分析停滞趋势。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.8}, 10))
        mem.add(ExperimentRecord(1, "GCN", {}, {"val_accuracy": 0.8}, 10))
        mem.add(ExperimentRecord(2, "GCN", {}, {"val_accuracy": 0.8}, 10))

        result = analyzer.analyze(mem, "classification")
        assert result["trend"] in ["plateau", "insufficient_data"]

    def test_analyze_declining(self):
        """测试分析下降趋势。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        # 添加更多数据点
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.8}, 10))
        mem.add(ExperimentRecord(1, "GCN", {}, {"val_accuracy": 0.78}, 10))
        mem.add(ExperimentRecord(2, "GCN", {}, {"val_accuracy": 0.76}, 10))
        mem.add(ExperimentRecord(3, "GCN", {}, {"val_accuracy": 0.74}, 10))
        mem.add(ExperimentRecord(4, "GCN", {}, {"val_accuracy": 0.72}, 10))
        mem.add(ExperimentRecord(5, "GCN", {}, {"val_accuracy": 0.7}, 10))

        result = analyzer.analyze(mem, "classification")
        assert result["trend"] in ["declining", "overfitting", "unstable"]

    def test_analyze_recommendation(self):
        """测试分析推荐任务。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "BPR", {}, {"ndcg@k": 0.1}, 10))
        mem.add(ExperimentRecord(1, "BPR", {}, {"ndcg@k": 0.12}, 10))

        result = analyzer.analyze(mem, "recommendation")
        assert "trend" in result
        assert "suggestions" in result

    def test_has_metric_delta(self):
        """测试包含指标变化。"""
        analyzer = FeedbackAnalyzer()
        mem = ExperimentMemory()
        mem.add(ExperimentRecord(0, "GCN", {}, {"val_accuracy": 0.7}, 10))
        mem.add(ExperimentRecord(1, "GCN", {}, {"val_accuracy": 0.8}, 10))

        result = analyzer.analyze(mem, "classification")
        assert "metric_delta" in result
        assert "variance" in result
        assert "improvement_rate" in result

"""测试 LLM 模块。"""

import pytest

from llm.schemas import ReflectionResult, PlannerDecision, StopDecision
from llm.parser import parse_structured, extract_json


class TestSchemas:
    """测试 Pydantic Schema。"""

    def test_reflection_result(self):
        """测试 ReflectionResult。"""
        result = ReflectionResult(
            observation="test",
            reasoning="test",
            next_action="run_baseline",
            confidence=0.9,
        )
        assert result.observation == "test"
        assert result.confidence == 0.9

    def test_planner_decision(self):
        """测试 PlannerDecision。"""
        decision = PlannerDecision(
            selected_config={"model_type": "GCN"},
            reason="test",
            confidence=0.8,
        )
        assert decision.selected_config["model_type"] == "GCN"

    def test_stop_decision(self):
        """测试 StopDecision。"""
        decision = StopDecision(
            should_stop=True,
            reason="budget exhausted",
        )
        assert decision.should_stop is True


class TestParser:
    """测试输出解析器。"""

    def test_parse_valid_json(self):
        """测试解析有效 JSON。"""
        text = '{"observation": "test", "reasoning": "test", "next_action": "run_baseline", "confidence": 0.9}'
        result = parse_structured(text, ReflectionResult)
        assert result is not None
        assert result.observation == "test"

    def test_parse_json_in_code_block(self):
        """测试解析代码块中的 JSON。"""
        text = '''```json
{
    "observation": "test",
    "reasoning": "test",
    "next_action": "run_baseline",
    "confidence": 0.9
}
```'''
        result = parse_structured(text, ReflectionResult)
        assert result is not None

    def test_parse_invalid_json(self):
        """测试解析无效 JSON。"""
        text = "This is not JSON"
        result = parse_structured(text, ReflectionResult)
        assert result is None

    def test_extract_json(self):
        """测试提取 JSON 字典。"""
        text = '{"key": "value"}'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_from_text(self):
        """测试从文本中提取 JSON。"""
        text = 'Some text {"key": "value"} more text'
        result = extract_json(text)
        assert result == {"key": "value"}

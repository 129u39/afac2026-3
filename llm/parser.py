"""LLM 输出解析器：从响应文本中提取结构化数据。"""

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_structured(response_text: str, schema_class: Type[T]) -> T | None:
    """从 LLM 响应文本中解析结构化输出。

    尝试多种方式提取 JSON：
    1. 直接解析整个响应
    2. 提取 ```json ... ``` 代码块
    3. 提取 { ... } JSON 对象

    Args:
        response_text: LLM 响应文本
        schema_class: Pydantic 模型类

    返回:
        解析后的 Pydantic 对象，失败返回 None
    """
    if not response_text:
        return None

    # 方式1：直接解析
    result = _try_parse(response_text, schema_class)
    if result:
        return result

    # 方式2：提取 ```json ... ``` 代码块
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if json_block:
        result = _try_parse(json_block.group(1).strip(), schema_class)
        if result:
            return result

    # 方式3：提取 { ... } JSON 对象
    json_obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
    if json_obj:
        result = _try_parse(json_obj.group(0), schema_class)
        if result:
            return result

    return None


def _try_parse(text: str, schema_class: Type[T]) -> T | None:
    """尝试将文本解析为 Pydantic 对象。"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return schema_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError, Exception):
        pass
    return None


def extract_json(response_text: str) -> dict | None:
    """从响应文本中提取 JSON 字典。

    Args:
        response_text: LLM 响应文本

    返回:
        解析后的字典，失败返回 None
    """
    if not response_text:
        return None

    # 直接解析
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 提取代码块
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # 提取 JSON 对象
    json_obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
    if json_obj:
        try:
            data = json.loads(json_obj.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None

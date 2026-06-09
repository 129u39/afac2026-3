"""Qwen 客户端：封装 DashScope OpenAI 兼容接口。"""

import os

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class QwenClient:
    """Qwen LLM 客户端。

    使用 DashScope 的 OpenAI 兼容接口调用 Qwen 模型。
    """

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str | None = None, model: str = "qwen-plus"):
        """
        Args:
            api_key: DashScope API Key，None 时从环境变量读取
            model: 模型名称
        """
        if not HAS_OPENAI:
            raise ImportError("openai is required. Install with: uv pip install openai")

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self._client = None

    @property
    def client(self) -> "OpenAI":
        """懒加载 OpenAI 客户端。"""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. "
                    "Set environment variable or pass api_key to QwenClient."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
            )
        return self._client

    @property
    def available(self) -> bool:
        """检查客户端是否可用。"""
        return bool(self.api_key) and HAS_OPENAI

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """发送聊天请求。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            model: 模型名称（覆盖默认）
            temperature: 温度
            max_tokens: 最大输出 token 数

        返回:
            模型响应文本
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

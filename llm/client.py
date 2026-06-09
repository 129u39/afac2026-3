"""Qwen 客户端：封装 DashScope SDK 接口。"""

import os
from pathlib import Path

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

try:
    from dashscope import Generation
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False


class QwenClient:
    """Qwen LLM 客户端。

    使用 DashScope SDK 调用 Qwen 模型。
    """

    def __init__(self, api_key: str | None = None, model: str = "qwen-plus"):
        """
        Args:
            api_key: DashScope API Key，None 时从环境变量读取
            model: 模型名称
        """
        if not HAS_DASHSCOPE:
            raise ImportError("dashscope is required. Install with: uv pip install dashscope")

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model

        # 设置 API Key 和 URL
        if self.api_key:
            dashscope.api_key = self.api_key
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    @property
    def available(self) -> bool:
        """检查客户端是否可用。"""
        return bool(self.api_key) and HAS_DASHSCOPE

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = Generation.call(
            api_key=self.api_key,
            model=model or self.model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 处理编码问题
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
        else:
            raise RuntimeError(
                f"DashScope API error: {response.status_code} "
                f"code={response.code} message={response.message}"
            )

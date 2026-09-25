"""OpenAI 兼容生成器（可选 SOTA 后端）。

通过 ``openai`` Python SDK 调用任意兼容 ``/v1/chat/completions`` 的服务
（OpenAI / vLLM / Ollama 等）。``api_key`` 与 ``base_url`` 默认从环境变量
读取（``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``），因此可由 ``RAGConfig`` 配置。
该依赖以 ``importorskip`` 隔离。
"""

from __future__ import annotations

import os
from typing import List, Optional

from ragnext.core.errors import GenerationError
from ragnext.core.types import Chunk
from ragnext.generation.base import Generator

try:  # pragma: no cover - 取决于可选依赖是否安装
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

DEFAULT_TEMPLATE = (
    "你是一个严谨的问答助手。请仅基于下面的「参考资料」回答问题，"
    "不要编造资料之外的信息。\n\n"
    "【参考资料】\n{context}\n\n"
    "【问题】\n{query}\n\n"
    "【回答】"
)


class OpenAICompatibleGenerator(Generator):
    """OpenAI 兼容 API 生成器。

    Args:
        model: 模型名（OpenAI / vLLM / Ollama 暴露的名称）。
        api_key: API Key；留空则从环境变量读取。
        base_url: API Base URL；留空则从环境变量读取（默认官方地址）。
        max_tokens: 生成上限。
        temperature: 采样温度（默认 0，结果更确定）。
        prompt_template: 自定义模板，需包含 ``{query}`` 与 ``{context}``。

    Raises:
        GenerationError (E600): ``openai`` 未安装或调用异常。
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
    ) -> None:
        if not _OPENAI_AVAILABLE or OpenAI is None:
            raise GenerationError(
                "openai 未安装，无法使用 OpenAI 兼容生成器：pip install openai",
                code="E600",
            )
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env, "")
        self.base_url = (
            base_url if base_url is not None else os.getenv(base_url_env, "https://api.openai.com/v1")
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template or DEFAULT_TEMPLATE
        try:
            self._client: OpenAI = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as exc:  # pragma: no cover - 客户端初始化异常
            raise GenerationError(f"初始化 OpenAI 客户端失败: {exc}", code="E600") from exc

    def generate(self, query: str, contexts: List[Chunk]) -> str:
        if not contexts:
            raise GenerationError("生成缺少上下文（contexts 为空）", code="E600")
        context_text = "\n\n".join(
            f"[来源 {i + 1}] {c.text}" for i, c in enumerate(contexts)
        )
        prompt = self.prompt_template.format(query=query, context=context_text)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # pragma: no cover - 网络/服务异常
            raise GenerationError(f"LLM 生成失败: {exc}", code="E600") from exc
        content = resp.choices[0].message.content
        return (content or "").strip()

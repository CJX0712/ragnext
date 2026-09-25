"""抽取式生成器（离线兜底，免 LLM、零密钥）。

从检索到的上下文中切出句子，按与查询的词元重叠度打分，挑出最相关的若干句
拼装成答案。无任何外部模型依赖，保证 demo 零手工干预跑通。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ragnext.core.errors import GenerationError
from ragnext.core.tokenizer import tokenize as _tokenize
from ragnext.core.types import Chunk
from ragnext.generation.base import Generator

# 句子切分：以中英文句末标点 / 分号 / 换行为断点。
_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _token_set(text: str) -> set:
    return set(_tokenize(text))


class ExtractiveGenerator(Generator):
    """抽取式生成器。

    Args:
        max_sentences: 答案最多拼装的句子数。
        max_chars: 答案字符上限；超过则提前截断。
    """

    def __init__(self, max_sentences: int = 3, max_chars: int = 300) -> None:
        if max_sentences <= 0:
            raise GenerationError(f"max_sentences 必须为正数: {max_sentences}", code="E600")
        if max_chars <= 0:
            raise GenerationError(f"max_chars 必须为正数: {max_chars}", code="E600")
        self.max_sentences = max_sentences
        self.max_chars = max_chars

    def generate(self, query: str, contexts: List[Chunk]) -> str:
        if not contexts:
            raise GenerationError("生成缺少上下文（contexts 为空）", code="E600")

        q_tokens = _token_set(query)
        scored: List[Tuple[float, str]] = []
        for chunk in contexts:
            for sent in _sentences(chunk.text):
                s_tokens = _token_set(sent)
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    continue
                # 重叠词数 / (句长 + 1)，兼顾相关性与简洁度。
                score = overlap / (len(s_tokens) + 1)
                scored.append((score, sent))

        if not scored:
            # 兜底：无词元重叠时，返回首个上下文的前若干句。
            first = contexts[0]
            fallback = _sentences(first.text)[: self.max_sentences]
            return "".join(fallback) if fallback else first.text[: self.max_chars]

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen: List[str] = []
        total = 0
        for _, sent in scored:
            if len(chosen) >= self.max_sentences:
                break
            if total + len(sent) > self.max_chars:
                break
            chosen.append(sent)
            total += len(sent)
        return "".join(chosen)

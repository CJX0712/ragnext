"""生成器基类，兼容 :class:`ragnext.core.interfaces.Generator` Protocol。"""

from __future__ import annotations

from typing import List

from ragnext.core.types import Chunk


class Generator:
    """所有生成器的抽象基类。"""

    def generate(self, query: str, contexts: List[Chunk]) -> str:
        """结合查询与上下文生成答案文本。"""
        raise NotImplementedError

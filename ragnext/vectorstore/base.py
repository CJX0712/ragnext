"""向量库基类，兼容 :class:`ragnext.core.interfaces.VectorStore` Protocol。"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ragnext.core.types import SearchResult


class VectorStore:
    """所有向量库的抽象基类，定义统一接口。"""

    def add(
        self, vectors: np.ndarray, ids: List[str], payloads: List[Any]
    ) -> None:
        """写入向量、标识与负载。"""
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """返回与查询向量最相近的 ``k`` 条命中。"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """持久化索引。"""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """载入索引。"""
        raise NotImplementedError

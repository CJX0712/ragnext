"""RAGNext 模块间契约（Protocol）。

``RAGPipeline`` 仅依赖以下 Protocol，不依赖任何具体实现，从而做到
实现可替换、模块可独立验证。所有 Protocol 均使用 ``runtime_checkable``，
便于在测试中做结构校验。
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

from ragnext.core.types import Chunk, QueryResult, SearchResult


@runtime_checkable
class Embedder(Protocol):
    """向量化器协议。

    Attributes:
        dim: 输出向量维度（实现类需以实例属性形式暴露）。
    """

    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为形状 ``(N, dim)`` 的向量矩阵。"""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库协议。"""

    def add(
        self, vectors: np.ndarray, ids: List[str], payloads: List[Any]
    ) -> None:
        """写入向量、标识与负载。"""
        ...

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """返回与查询向量最相近的 ``k`` 条命中（分数越大越相关）。"""
        ...

    def save(self, path: str) -> None:
        """将索引持久化到 ``path``。"""
        ...

    def load(self, path: str) -> None:
        """从 ``path`` 载入索引。"""
        ...


@runtime_checkable
class Retriever(Protocol):
    """检索器协议。"""

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """根据查询串返回最相关的 ``k`` 个文本块。"""
        ...


@runtime_checkable
class Generator(Protocol):
    """生成器协议。"""

    def generate(self, query: str, contexts: List[Chunk]) -> str:
        """结合查询与上下文生成答案文本。"""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """评测器协议。"""

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[Chunk],
        ground_truth: str | None = None,
    ) -> Dict[str, float]:
        """返回以指标名为键、数值为值的评测字典。"""
        ...

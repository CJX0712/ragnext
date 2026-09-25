"""检索器：向量库 TopK 召回 + 可选 CrossEncoder 重排。

``TopKRetriever`` 通过嵌入器将查询编码为向量，调用向量库 ``search`` 召回
``k`` 个最相关块；若配置 ``use_rerank`` 且提供了 ``reranker``，则再用
CrossEncoder 对候选块做精排。空查询或检索为空时抛出 ``E500``。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ragnext.core.errors import RetrievalError
from ragnext.core.types import Chunk, SearchResult
from ragnext.core.interfaces import Embedder, VectorStore

try:  # pragma: no cover - 取决于可选依赖是否安装
    from sentence_transformers import CrossEncoder

    _CE_AVAILABLE = True
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]
    _CE_AVAILABLE = False


class TopKRetriever:
    """基于向量库的 TopK 检索器。

    Args:
        embedder: 实现 ``Embedder`` 协议的向量化器。
        vector_store: 实现 ``VectorStore`` 协议的向量库。
        top_k: 默认召回数量。
        use_rerank: 是否启用重排。
        reranker: 可选的重排器实例（实现 ``rerank(query, chunks, k)``）。
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 4,
        use_rerank: bool = False,
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.reranker = reranker

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        """返回与查询最相关的 ``k`` 个 :class:`Chunk`（已按相关性排序）。"""
        if not query or not query.strip():
            raise RetrievalError("查询为空，无法检索", code="E500")
        k = k or self.top_k
        q_vec = self.embedder.embed([query])[0]
        results: List[SearchResult] = self.vector_store.search(q_vec, k)
        if not results:
            raise RetrievalError("检索结果为空（索引可能为空）", code="E500")

        chunks = [r.payload for r in results if isinstance(r.payload, Chunk)]
        if self.use_rerank and self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks, k)
        return chunks


class CrossEncoderReranker:
    """CrossEncoder 重排器（可选 SOTA 后端）。

    Args:
        model_name: CrossEncoder 模型名，默认 ``cross-encoder/ms-marco-MiniLM-L-6-v2``。

    Raises:
        RetrievalError (E500): ``sentence-transformers`` 未安装或模型加载失败。
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        if not _CE_AVAILABLE or CrossEncoder is None:
            raise RetrievalError(
                "sentence-transformers 未安装，无法使用 CrossEncoder 重排："
                "pip install sentence-transformers",
                code="E500",
            )
        self.model_name = model_name
        try:
            self._model: CrossEncoder = CrossEncoder(model_name)
        except Exception as exc:  # pragma: no cover - 网络/权重异常
            raise RetrievalError(
                f"加载 CrossEncoder 失败 ({model_name}): {exc}", code="E500"
            ) from exc

    def rerank(self, query: str, chunks: List[Chunk], k: int) -> List[Chunk]:
        """按 CrossEncoder 相关性分数对候选块重排，返回前 ``k`` 个。"""
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = np.asarray(self._model.predict(pairs))
        order = np.argsort(-scores)
        return [chunks[int(i)] for i in order[: min(k, len(chunks))]]

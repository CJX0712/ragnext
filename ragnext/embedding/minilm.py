"""MiniLM 稠密嵌入器（可选 SOTA 后端）。

封装 ``sentence-transformers`` 的 ``all-MiniLM-L6-v2``。该依赖体积较大且需
联网下载权重，因此以 ``importorskip`` 隔离：模块可被安全导入，但仅当
``sentence-transformers`` 已安装、且权重可获取时才能构造实例，否则抛出
``E300``（:class:`EmbeddingError`）。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ragnext.embedding.base import BaseEmbedder
from ragnext.core.errors import EmbeddingError

try:  # pragma: no cover - 取决于可选依赖是否安装
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _ST_AVAILABLE = False


class MiniLMEmbedder(BaseEmbedder):
    """基于 ``sentence-transformers`` 的稠密嵌入器。

    Args:
        model_name: 模型名，默认 ``all-MiniLM-L6-v2``（384 维）。

    Raises:
        EmbeddingError (E300): ``sentence-transformers`` 未安装或权重加载失败。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _ST_AVAILABLE or SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers 未安装，无法使用 MiniLM 嵌入器："
                "pip install sentence-transformers",
                code="E300",
            )
        self.model_name = model_name
        try:
            self._model: SentenceTransformer = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - 网络/权重异常
            raise EmbeddingError(
                f"加载 MiniLM 模型失败 ({model_name}): {exc}", code="E300"
            ) from exc
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        """编码文本列表为 ``(N, dim)`` 的 L2 归一化矩阵。"""
        if not isinstance(texts, (list, tuple)):
            raise EmbeddingError("texts 必须为序列", code="E300")
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        vectors = self._model.encode(
            [str(t) for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float64)

"""Numpy 精确向量库（离线兜底，零依赖、可持久化）。

采用 numpy 做**精确**相似度检索（cosine / l2），保证结果可复现、可单测。
索引通过 ``numpy`` 的 ``.npz``（``allow_pickle``）持久化，能保存任意 Python
对象负载（如 ``Chunk`` 实例）。
"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np

from ragnext.core.errors import VectorStoreError
from ragnext.core.types import SearchResult
from ragnext.vectorstore.base import VectorStore


class NumpyVectorStore(VectorStore):
    """基于 numpy 的精确向量库。

    Args:
        metric: ``"cosine"``（默认）或 ``"l2"``。两种度量均统一为
            “分数越大越相关”。

    Raises:
        VectorStoreError (E400): 非法度量、维度不一致或 IO 失败。
    """

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in ("cosine", "l2"):
            raise VectorStoreError(
                f"不支持的 metric: {metric}（应为 cosine/l2）", code="E400"
            )
        self.metric = metric
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._ids: List[str] = []
        self._payloads: List[Any] = []
        self._built = False

    # ------------------------------------------------------------------
    def add(
        self, vectors: np.ndarray, ids: List[str], payloads: List[Any]
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise VectorStoreError(
                f"vectors 必须为 2D，实际 {vectors.ndim}D", code="E400"
            )
        if not (len(ids) == vectors.shape[0] == len(payloads)):
            raise VectorStoreError(
                "ids / vectors / payloads 长度不一致", code="E400"
            )
        if self._built and vectors.shape[1] != self._vectors.shape[1]:
            raise VectorStoreError(
                f"维度不匹配：已有 {self._vectors.shape[1]}，新增 {vectors.shape[1]}",
                code="E400",
            )
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        if self._built:
            self._vectors = np.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors
            self._built = True
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        if not self._built:
            raise VectorStoreError("向量库未构建，请先调用 add()", code="E400")
        if k <= 0:
            raise VectorStoreError(f"k 必须为正数: {k}", code="E400")

        q = np.asarray(query_vector, dtype=np.float64).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise VectorStoreError(
                f"查询维度({q.shape[0]})与库维度({self._vectors.shape[1]})不一致",
                code="E400",
            )
        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
            if norm > 0.0:
                q = q / norm
            scores = self._vectors @ q
        else:  # l2
            diff = self._vectors - q
            scores = -np.linalg.norm(diff, axis=1)

        k = min(k, len(self._ids))
        order = np.argsort(-scores)[:k]
        return [
            SearchResult(
                id=self._ids[i],
                score=float(scores[i]),
                payload=self._payloads[i],
            )
            for i in order
        ]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self._built:
            raise VectorStoreError("空向量库不可保存", code="E400")
        np.savez(
            path,
            vectors=self._vectors,
            ids=np.array(self._ids, dtype=object),
            payloads=np.array(self._payloads, dtype=object),
            metric=np.array([self.metric], dtype=object),
        )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise VectorStoreError(f"索引文件不存在: {path}", code="E400")
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as exc:  # pragma: no cover - 损坏文件
            raise VectorStoreError(f"读取索引失败: {path}: {exc}", code="E400") from exc
        self._vectors = np.asarray(data["vectors"], dtype=np.float64)
        self._ids = [str(x) for x in list(data["ids"])]
        self._payloads = list(data["payloads"])
        self.metric = str(data["metric"][0])
        self._built = True

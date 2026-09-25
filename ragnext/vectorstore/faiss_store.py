"""FAISS 向量库（可选 SOTA 后端）。

复用 Meta 的 ``faiss-cpu``（``IndexFlatIP`` / ``IndexFlatL2``）做工业级 ANN
检索。由于 faiss 在部分环境装包存在风险，本模块以 ``importorskip`` 隔离：
可被安全导入；仅在 ``faiss-cpu`` 安装后构造实例才可用，否则抛出 ``E400``。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from ragnext.core.errors import VectorStoreError
from ragnext.core.types import SearchResult
from ragnext.vectorstore.base import VectorStore

try:  # pragma: no cover - 取决于可选依赖是否安装
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False


class FAISSVectorStore(VectorStore):
    """基于 FAISS 的向量库。

    Args:
        metric: ``"cosine"``（默认，内部用 ``IndexFlatIP``）或 ``"l2"``。
        dim: 向量维度；留空则在首次 ``add`` 时自动推断。

    Raises:
        VectorStoreError (E400): ``faiss-cpu`` 未安装或 IO 失败。
    """

    def __init__(self, metric: str = "cosine", dim: Optional[int] = None) -> None:
        if not _FAISS_AVAILABLE or faiss is None:
            raise VectorStoreError(
                "faiss-cpu 未安装，无法使用 FAISS 向量库：pip install faiss-cpu",
                code="E400",
            )
        if metric not in ("cosine", "l2"):
            raise VectorStoreError(
                f"不支持的 metric: {metric}（应为 cosine/l2）", code="E400"
            )
        self.metric = metric
        self.dim: Optional[int] = dim
        self._index = None
        self._ids: List[str] = []
        self._payloads: List[Any] = []
        self._built = False

    # ------------------------------------------------------------------
    def add(
        self, vectors: np.ndarray, ids: List[str], payloads: List[Any]
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise VectorStoreError(
                f"vectors 必须为 2D，实际 {vectors.ndim}D", code="E400"
            )
        if not (len(ids) == vectors.shape[0] == len(payloads)):
            raise VectorStoreError(
                "ids / vectors / payloads 长度不一致", code="E400"
            )
        dim = vectors.shape[1]
        if self._index is None:
            self.dim = dim
            self._index = (
                faiss.IndexFlatIP(dim)
                if self.metric == "cosine"
                else faiss.IndexFlatL2(dim)
            )
        else:
            if dim != self.dim:
                raise VectorStoreError(
                    f"维度不匹配：已有 {self.dim}，新增 {dim}", code="E400"
                )

        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        self._index.add(vectors)
        self._ids.extend(ids)
        self._payloads.extend(payloads)
        self._built = True

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        if not self._built or self._index is None:
            raise VectorStoreError("向量库未构建，请先调用 add()", code="E400")
        if k <= 0:
            raise VectorStoreError(f"k 必须为正数: {k}", code="E400")

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
            if norm > 0.0:
                q = q / norm
            scores, idxs = self._index.search(q, k)
        else:  # l2
            dist, idxs = self._index.search(q, k)
            scores = -dist

        k = min(k, len(self._ids))
        results: List[SearchResult] = []
        for j in range(min(k, idxs.shape[1])):
            i = int(idxs[0][j])
            if i == -1:
                continue
            results.append(
                SearchResult(
                    id=self._ids[i],
                    score=float(scores[0][j]),
                    payload=self._payloads[i],
                )
            )
        return results

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self._built or self._index is None:
            raise VectorStoreError("空向量库不可保存", code="E400")
        assert faiss is not None
        faiss.write_index(self._index, path)
        meta_path = self._meta_path(path)
        np.savez(
            meta_path,
            ids=np.array(self._ids, dtype=object),
            payloads=np.array(self._payloads, dtype=object),
            metric=np.array([self.metric], dtype=object),
        )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise VectorStoreError(f"索引文件不存在: {path}", code="E400")
        assert faiss is not None
        try:
            self._index = faiss.read_index(path)
        except Exception as exc:  # pragma: no cover - 损坏文件
            raise VectorStoreError(f"读取 FAISS 索引失败: {path}: {exc}", code="E400") from exc
        self.dim = int(self._index.d)
        meta = np.load(self._meta_path(path), allow_pickle=True)
        self._ids = [str(x) for x in list(meta["ids"])]
        self._payloads = list(meta["payloads"])
        self.metric = str(meta["metric"][0])
        self._built = True

    @staticmethod
    def _meta_path(index_path: str) -> str:
        """FAISS 索引的元数据（ids/payloads）并存文件路径。"""
        base, _ = os.path.splitext(index_path)
        return base + ".meta.npz"

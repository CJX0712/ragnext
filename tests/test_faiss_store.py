"""vectorstore.faiss_store FAISS 向量库测试（可选后端）。

若环境中未安装 ``faiss-cpu``，本模块整体被 ``pytest.importorskip`` 跳过，
不影响离线主线的单测全绿。
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from ragnext.core.errors import VectorStoreError
from ragnext.core.types import Chunk, SearchResult
from ragnext.vectorstore.faiss_store import FAISSVectorStore


def _vecs():
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)


def _ids():
    return ["a", "b", "c"]


def _payloads():
    return [Chunk(id=x, doc_id="d", text=x) for x in ["a", "b", "c"]]


def test_add_search_cosine():
    store = FAISSVectorStore(metric="cosine")
    store.add(_vecs(), _ids(), _payloads())
    res = store.search(np.array([1.0, 0, 0], dtype=np.float32), k=2)
    assert isinstance(res[0], SearchResult)
    assert res[0].id == "a"
    assert len(res) == 2


def test_search_not_built():
    store = FAISSVectorStore()
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(2, dtype=np.float32), k=1)


def test_save_load(tmp_path):
    store = FAISSVectorStore(metric="cosine")
    store.add(_vecs(), _ids(), _payloads())
    p = tmp_path / "idx.faiss"
    store.save(str(p))
    store2 = FAISSVectorStore()
    store2.load(str(p))
    res = store2.search(np.array([0, 1.0, 0], dtype=np.float32), k=1)
    assert res[0].id == "b"
    assert isinstance(res[0].payload, Chunk)


def test_dim_mismatch():
    store = FAISSVectorStore()
    store.add(_vecs(), _ids(), _payloads())
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((1, 5), dtype=np.float32), ["x"], [None])


def test_invalid_metric():
    with pytest.raises(VectorStoreError):
        FAISSVectorStore(metric="hamming")


def test_l2_metric():
    store = FAISSVectorStore(metric="l2")
    store.add(_vecs(), _ids(), _payloads())
    res = store.search(np.array([0, 0, 1.0], dtype=np.float32), k=1)
    assert res[0].id == "c"

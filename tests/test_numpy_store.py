"""vectorstore.numpy_store 精确向量库测试。"""

import numpy as np
import pytest

from ragnext.core.errors import VectorStoreError
from ragnext.core.types import Chunk, SearchResult
from ragnext.vectorstore.numpy_store import NumpyVectorStore


def _vecs():
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float64)


def _ids():
    return ["a", "b", "c"]


def _payloads():
    return [Chunk(id=x, doc_id="d", text=x) for x in ["a", "b", "c"]]


def test_add_search_cosine():
    store = NumpyVectorStore(metric="cosine")
    store.add(_vecs(), _ids(), _payloads())
    res = store.search(np.array([1.0, 0, 0]), k=2)
    assert isinstance(res[0], SearchResult)
    assert res[0].id == "a"
    assert len(res) == 2


def test_search_not_built():
    store = NumpyVectorStore()
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(2), k=1)


def test_save_load(tmp_path):
    store = NumpyVectorStore(metric="cosine")
    store.add(_vecs(), _ids(), _payloads())
    p = tmp_path / "idx.npz"
    store.save(str(p))
    store2 = NumpyVectorStore()
    store2.load(str(p))
    res = store2.search(np.array([0, 1.0, 0]), k=1)
    assert res[0].id == "b"
    assert isinstance(res[0].payload, Chunk)


def test_dim_mismatch():
    store = NumpyVectorStore()
    store.add(_vecs(), _ids(), _payloads())
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((1, 5)), ["x"], [None])


def test_invalid_metric():
    with pytest.raises(VectorStoreError):
        NumpyVectorStore(metric="hamming")


def test_l2_metric():
    store = NumpyVectorStore(metric="l2")
    store.add(_vecs(), _ids(), _payloads())
    res = store.search(np.array([0, 0, 1.0]), k=1)
    assert res[0].id == "c"


def test_load_missing_file():
    store = NumpyVectorStore()
    with pytest.raises(VectorStoreError):
        store.load("/no/such/index.npz")

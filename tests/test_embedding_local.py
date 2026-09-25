"""embedding.local 本地哈希嵌入器测试。"""

import numpy as np
import pytest

from ragnext.core.errors import EmbeddingError
from ragnext.embedding.local import LocalHashingEmbedder


def test_dim():
    emb = LocalHashingEmbedder(dim=128)
    assert emb.dim == 128


def test_shape_and_determinism():
    emb = LocalHashingEmbedder(dim=64)
    v1 = emb.embed(["密码重置"])
    v2 = emb.embed(["密码重置"])
    assert v1.shape == (1, 64)
    assert np.allclose(v1, v2)


def test_different_texts_differ():
    emb = LocalHashingEmbedder(dim=64)
    a = emb.embed(["如何重置密码"])
    b = emb.embed(["今天天气真好"])
    assert not np.allclose(a, b)


def test_normalization():
    emb = LocalHashingEmbedder(dim=32)
    v = emb.embed(["测试文本"])
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-9 or norm == 0.0


def test_empty_list():
    emb = LocalHashingEmbedder(dim=16)
    v = emb.embed([])
    assert v.shape == (0, 16)


def test_invalid_dim():
    with pytest.raises(EmbeddingError):
        LocalHashingEmbedder(dim=0)


def test_multi_text_shape():
    emb = LocalHashingEmbedder(dim=48)
    v = emb.embed(["a", "b", "c"])
    assert v.shape == (3, 48)

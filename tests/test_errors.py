"""core.errors 异常层级与错误码测试。"""

import pytest

from ragnext.core.errors import (
    ChunkingError,
    EmbeddingError,
    EvalError,
    GenerationError,
    IngestionError,
    RAGNextError,
    RetrievalError,
    VectorStoreError,
)


def test_codes():
    assert IngestionError().code == "E100"
    assert ChunkingError().code == "E200"
    assert EmbeddingError().code == "E300"
    assert VectorStoreError().code == "E400"
    assert RetrievalError().code == "E500"
    assert GenerationError().code == "E600"
    assert EvalError().code == "E700"


def test_hierarchy():
    for cls in (
        IngestionError,
        ChunkingError,
        EmbeddingError,
        VectorStoreError,
        RetrievalError,
        GenerationError,
        EvalError,
    ):
        assert issubclass(cls, RAGNextError)


def test_message_format():
    err = IngestionError("boom")
    assert "[E100]" in str(err)
    assert "boom" in str(err)


def test_custom_code():
    err = RAGNextError("custom", code="E999")
    assert err.code == "E999"
    assert "[E999]" in str(err)


def test_raise_propagation():
    with pytest.raises(RAGNextError):
        raise ChunkingError("x")

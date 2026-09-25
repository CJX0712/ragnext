"""retrieval.retriever 检索器测试。"""

import pytest

from ragnext.chunking.splitter import RecursiveCharacterSplitter
from ragnext.core.errors import RetrievalError
from ragnext.core.types import Chunk, Document
from ragnext.embedding.local import LocalHashingEmbedder
from ragnext.retrieval.retriever import TopKRetriever
from ragnext.vectorstore.numpy_store import NumpyVectorStore


def _build():
    docs = [
        Document(id="d1", text="如何重置密码，点击忘记密码并输入验证码完成验证。"),
        Document(id="d2", text="我们支持微信支付和支付宝两种移动支付方式。"),
    ]
    sp = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=5)
    chunks = []
    for d in docs:
        chunks.extend(sp.split(d))
    emb = LocalHashingEmbedder(dim=64)
    store = NumpyVectorStore()
    store.add(emb.embed([c.text for c in chunks]), [c.id for c in chunks], chunks)
    return TopKRetriever(emb, store, top_k=2), chunks


def test_retrieve_returns_chunks():
    retriever, _ = _build()
    out = retriever.retrieve("如何重置密码")
    assert all(isinstance(c, Chunk) for c in out)
    assert len(out) <= 2


def test_empty_query_raises():
    retriever, _ = _build()
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")


def test_retrieve_relevant_first():
    retriever, _ = _build()
    out = retriever.retrieve("如何重置密码")
    assert out[0].doc_id == "d1"


def test_retrieve_payment_query():
    retriever, _ = _build()
    out = retriever.retrieve("支持哪些支付方式")
    assert out[0].doc_id == "d2"

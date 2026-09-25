"""core.types 数据结构测试。"""

from ragnext.core.types import Chunk, Document, QueryResult, SearchResult


def test_document_defaults():
    doc = Document(id="1", text="hello")
    assert doc.id == "1"
    assert doc.text == "hello"
    assert doc.metadata == {}


def test_chunk_fields():
    chunk = Chunk(id="c1", doc_id="d1", text="x", metadata={"i": 1})
    assert chunk.doc_id == "d1"
    assert chunk.metadata["i"] == 1


def test_search_result_payload():
    res = SearchResult(id="c1", score=0.9, payload="payload")
    assert res.score == 0.9
    assert res.payload == "payload"


def test_query_result_defaults():
    q = QueryResult(query="q", answer="a")
    assert q.contexts == []
    assert q.latency_ms == 0.0

"""chunking.splitter 递归字符切分测试。"""

import pytest

from ragnext.chunking.splitter import RecursiveCharacterSplitter
from ragnext.core.errors import ChunkingError
from ragnext.core.types import Document


def test_split_basic():
    text = "这是第一句。这是第二句。这是第三句，内容稍长一些用来测试切分效果。"
    doc = Document(id="d", text=text)
    sp = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=4)
    chunks = sp.split(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "d"
        assert c.id.startswith("d-c")
    joined = "".join(c.text for c in chunks)
    assert "这是第一句" in joined


def test_split_empty_raises():
    doc = Document(id="d", text="   ")
    sp = RecursiveCharacterSplitter()
    with pytest.raises(ChunkingError) as exc:
        sp.split(doc)
    assert exc.value.code == "E200"


def test_overlap_invalid():
    with pytest.raises(ChunkingError):
        RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=10)


def test_chunk_size_positive():
    with pytest.raises(ChunkingError):
        RecursiveCharacterSplitter(chunk_size=0)


def test_overlap_preserved():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    doc = Document(id="d", text=text)
    sp = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=10)
    chunks = sp.split(doc)
    assert len(chunks) >= 2
    # 总字符数应 >= 原文（重叠区被重复计入）
    assert sum(len(c.text) for c in chunks) >= len(text)

"""ingestion.loaders 加载器测试（含 PDF 手写最小样本）。"""

from pathlib import Path

import pytest

from ragnext.core.errors import IngestionError
from ragnext.ingestion.loaders import (
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    load_document,
)


def _make_minimal_pdf(path: Path, text: str) -> None:
    """用标准库手写一个合法的最小 PDF（含 xref），供 PDFLoader 解析。"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = f"BT /F1 12 Tf 50 150 Td ({text}) Tj ET".encode("utf-8")
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    path.write_bytes(out)


def test_text_loader(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    doc = TextLoader().load(p)
    assert doc.text == "hello world"
    assert doc.metadata["type"] == "text"


def test_markdown_loader(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# title", encoding="utf-8")
    doc = load_document(p)
    assert doc.text == "# title"
    assert doc.metadata["type"] == "markdown"


def test_pdf_loader(tmp_path):
    p = tmp_path / "a.pdf"
    _make_minimal_pdf(p, "Hello PDF")
    doc = PDFLoader().load(p)
    assert "Hello" in doc.text
    assert doc.metadata["type"] == "pdf"


def test_missing_file_raises():
    with pytest.raises(IngestionError) as exc:
        load_document("/no/such/file.txt")
    assert exc.value.code == "E100"


def test_unsupported_type_raises(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(IngestionError) as exc:
        load_document(p)
    assert exc.value.code == "E100"

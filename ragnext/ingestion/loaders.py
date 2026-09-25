"""文档加载器：支持纯文本、Markdown 与 PDF。

对外提供 :func:`load_document` 按扩展名分发的统一入口；文件不存在或类型
不支持时抛出 ``E100``（:class:`IngestionError`）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ragnext.core.errors import IngestionError
from ragnext.core.types import Document

PathLike = Union[str, Path]


class BaseLoader:
    """加载器基类，子类实现 :meth:`load`。"""

    def load(self, path: PathLike) -> Document:
        """加载单个文件为 :class:`Document`。"""
        raise NotImplementedError


class TextLoader(BaseLoader):
    """纯文本（``.txt``）加载器。"""

    def load(self, path: PathLike) -> Document:
        p = Path(path)
        if not p.exists():
            raise IngestionError(f"文件不存在: {path}", code="E100")
        text = p.read_text(encoding="utf-8", errors="replace")
        return Document(
            id=p.stem,
            text=text,
            metadata={"source": str(p), "type": "text"},
        )


class MarkdownLoader(BaseLoader):
    """Markdown（``.md`` / ``.markdown``）加载器，内容与纯文本一致读取，
    仅元数据 ``type`` 不同，便于后续按块保留来源语义。"""

    def load(self, path: PathLike) -> Document:
        p = Path(path)
        if not p.exists():
            raise IngestionError(f"文件不存在: {path}", code="E100")
        text = p.read_text(encoding="utf-8", errors="replace")
        return Document(
            id=p.stem,
            text=text,
            metadata={"source": str(p), "type": "markdown"},
        )


class PDFLoader(BaseLoader):
    """PDF（``.pdf``）加载器，复用 ``pypdf`` 抽取每页文本。

    若 ``pypdf`` 未安装，构造/加载时抛出 ``E100`` 并提示安装方式。
    """

    def load(self, path: PathLike) -> Document:
        p = Path(path)
        if not p.exists():
            raise IngestionError(f"文件不存在: {path}", code="E100")
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 依赖缺失提示
            raise IngestionError(
                "pypdf 未安装，无法解析 PDF：pip install pypdf",
                code="E100",
            ) from exc
        reader = PdfReader(str(p))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pragma: no cover - 个别损坏页容错
                pages.append("")
        text = "\n".join(pages)
        return Document(
            id=p.stem,
            text=text,
            metadata={"source": str(p), "type": "pdf", "pages": len(reader.pages)},
        )


def load_document(path: PathLike) -> Document:
    """按文件扩展名分发到对应加载器，返回 :class:`Document`。

    Raises:
        IngestionError (E100): 文件不存在或扩展名不被支持。
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"文件不存在: {path}", code="E100")
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return TextLoader().load(p)
    if suffix in (".md", ".markdown"):
        return MarkdownLoader().load(p)
    if suffix == ".pdf":
        return PDFLoader().load(p)
    raise IngestionError(f"不支持的文件类型: {suffix} ({path})", code="E100")

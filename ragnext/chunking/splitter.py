"""递归字符切分器（Recursive Character Splitter）。

按分隔符优先级递归切分，再合并为不超过 ``chunk_size`` 的块，相邻块之间
保留 ``chunk_overlap`` 字符的重叠，以尽量减少跨块语义断裂。
空文本或非法参数抛出 ``E200``（:class:`ChunkingError`）。
"""

from __future__ import annotations

from typing import List

from ragnext.core.errors import ChunkingError
from ragnext.core.types import Chunk, Document

# 分隔符优先级：先按段落/句子，再按字符。覆盖中英文常见断点。
DEFAULT_SEPARATORS: List[str] = [
    "\n\n",
    "\n",
    "。",
    "；",
    "！",
    "？",
    ".",
    "!",
    "?",
    " ",
    "",
]


class RecursiveCharacterSplitter:
    """递归字符切分器。

    Args:
        chunk_size: 单块目标长度（字符）。
        chunk_overlap: 相邻块重叠长度（字符），必须严格小于 ``chunk_size``。
        separators: 自定义分隔符列表；留空使用 :data:`DEFAULT_SEPARATORS`。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size 必须为正数: {chunk_size}", code="E200")
        if chunk_overlap < 0:
            raise ChunkingError(
                f"chunk_overlap 不能为负: {chunk_overlap}", code="E200"
            )
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})",
                code="E200",
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators else list(DEFAULT_SEPARATORS)

    def split(self, doc: Document) -> List[Chunk]:
        """将文档切分为若干 :class:`Chunk`。

        Raises:
            ChunkingError (E200): 文档文本为空或仅含空白。
        """
        text = doc.text
        if not text or not text.strip():
            raise ChunkingError(f"无法切分空文本，doc_id={doc.id}", code="E200")
        pieces = self._split_recursive(text, self.separators)
        merged = self._merge_chunks(pieces)
        chunks: List[Chunk] = []
        for index, piece in enumerate(merged):
            chunks.append(
                Chunk(
                    id=f"{doc.id}-c{index}",
                    doc_id=doc.id,
                    text=piece,
                    metadata={**doc.metadata, "chunk_index": index},
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归地把文本切成各自小于 ``chunk_size`` 的片段（保留分隔符）。"""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        if separator == "":
            # 字符级兜底切分
            step = max(1, self.chunk_size)
            return [text[i : i + step] for i in range(0, len(text), step)]

        good_splits: List[str] = []
        for part in text.split(separator):
            if not part:
                continue
            good_splits.append(part + separator)
        if not good_splits:
            return [text]

        pieces: List[str] = []
        remaining = separators[separators.index(separator) + 1 :]
        for piece in good_splits:
            if len(piece) <= self.chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._split_recursive(piece, remaining))
        return pieces

    def _merge_chunks(self, pieces: List[str]) -> List[str]:
        """将过小的片段合并为接近 ``chunk_size`` 的块，并注入重叠。"""
        merged: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if not current:
                current = piece
            elif len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                merged.append(current)
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap :] + piece
                else:
                    current = piece
        if current:
            merged.append(current)
        return merged

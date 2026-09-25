"""RAGNext 共享数据结构定义。

所有跨模块流转的数据均使用标准 ``dataclass`` 描述，保证可序列化、可比较、
可独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """一条待索引的原始文档。

    Attributes:
        id: 文档唯一标识（通常由文件名或调用方指定）。
        text: 文档纯文本内容。
        metadata: 任意元数据，例如来源路径、文档类型等。
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """文档切分后的最小检索单元。

    Attributes:
        id: 块唯一标识，约定为 ``{doc_id}-c{index}``。
        doc_id: 所属文档的 ``Document.id``。
        text: 块文本。
        metadata: 继承并补充的元数据，例如 ``chunk_index``。
    """

    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """向量库单次检索返回的一条命中。

    Attributes:
        id: 命中项的标识（与写入向量库时一致）。
        score: 相似度分数，越大越相关（cosine/l2 均已统一为“越大越好”）。
        payload: 命中项关联的负载，通常是 ``Chunk`` 实例或字典。
    """

    id: str
    score: float
    payload: Any = None


@dataclass
class QueryResult:
    """一次完整 RAG 查询的结果。

    Attributes:
        query: 原始查询串。
        answer: 生成器产出的答案文本。
        contexts: 参与生成的检索上下文块列表（按相关性排序）。
        latency_ms: 端到端耗时（毫秒）。
    """

    query: str
    answer: str
    contexts: List[Chunk] = field(default_factory=list)
    latency_ms: float = 0.0

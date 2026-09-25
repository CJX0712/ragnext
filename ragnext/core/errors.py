"""RAGNext 统一异常层级。

所有业务异常均继承自 :class:`RAGNextError`，并携带稳定的 ``code`` 属性，
便于上层（CLI / 流水线）按错误码做透传与归类展示。

错误码区间（见设计文档第 4.1 节）：
    E100  IngestionError   加载失败
    E200  ChunkingError    空文本 / 非法参数
    E300  EmbeddingError   维度不匹配 / 模型缺失
    E400  VectorStoreError 未构建 / IO 失败
    E500  RetrievalError   空结果 / 索引缺失
    E600  GenerationError  生成失败
    E700  EvalError        指标非法
"""

from __future__ import annotations


class RAGNextError(Exception):
    """所有 RAGNext 异常的根类型。

    Args:
        message: 人类可读的错误描述。
        code: 可选，覆盖默认错误码（一般用于复用异常类时指定更精确码）。
    """

    code: str = "E000"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        if code is not None:
            self.code = code
        super().__init__(f"[{self.code}] {message}")

    def __str__(self) -> str:  # pragma: no cover - 纯展示
        return f"[{self.code}] {self.args[0]}"


class IngestionError(RAGNextError):
    """E100：文档加载失败（文件不存在、格式不支持等）。"""

    code = "E100"


class ChunkingError(RAGNextError):
    """E200：文本切分失败（空文本、非法 chunk 参数等）。"""

    code = "E200"


class EmbeddingError(RAGNextError):
    """E300：向量化失败（维度不匹配、模型缺失等）。"""

    code = "E300"


class VectorStoreError(RAGNextError):
    """E400：向量库失败（未构建、维度不一致、IO 错误等）。"""

    code = "E400"


class RetrievalError(RAGNextError):
    """E500：检索失败（空查询、索引缺失、无结果等）。"""

    code = "E500"


class GenerationError(RAGNextError):
    """E600：答案生成失败（无上下文、LLM 调用异常等）。"""

    code = "E600"


class EvalError(RAGNextError):
    """E700：评测指标非法（空 ground truth、除零等）。"""

    code = "E700"

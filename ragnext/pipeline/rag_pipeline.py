"""RAG 流水线：单向编排 Ingestion→Chunking→Embedding→VectorStore→
Retrieval→Generation，并负责错误码透传。

``RAGPipeline`` 仅依赖各模块的 Protocol（``Embedder`` / ``VectorStore`` /
``Retriever`` / ``Generator``），实现可插拔、可独立验证。典型用法：

    pipeline = RAGPipeline.from_config(RAGConfig(...))
    pipeline.index(["./docs/faq.txt"])
    result = pipeline.run("如何重置密码？")
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

from ragnext.core.config import RAGConfig
from ragnext.core.errors import (
    ChunkingError,
    GenerationError,
    IngestionError,
    RAGNextError,
)
from ragnext.core.interfaces import Embedder, Generator, Retriever, VectorStore
from ragnext.core.types import Chunk, Document, QueryResult


class RAGPipeline:
    """RAG 流水线编排器。

    Args:
        embedder: 嵌入器（实现 ``Embedder``）。
        vector_store: 向量库（实现 ``VectorStore``）。
        retriever: 检索器（实现 ``Retriever``）。
        generator: 生成器（实现 ``Generator``）。
        config: 可选运行配置，用于 ``run`` 的默认 ``top_k`` 等。
        evaluator: 可选评测器。
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: Retriever,
        generator: Generator,
        config: Optional[RAGConfig] = None,
        evaluator: Optional[object] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.config = config or RAGConfig()
        self.evaluator = evaluator
        self._chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    def index(self, documents: List[Union[str, Document]]) -> int:
        """索引文档（路径或 ``Document``）。返回生成的文本块数量。

        Raises:
            IngestionError (E100) / ChunkingError (E200): 加载或切分失败。
        """
        from ragnext.chunking.splitter import RecursiveCharacterSplitter
        from ragnext.ingestion.loaders import load_document

        splitter = RecursiveCharacterSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks: List[Chunk] = []
        for doc in documents:
            if isinstance(doc, Document):
                document = doc
            elif isinstance(doc, str):
                document = load_document(doc)
            else:
                raise IngestionError(
                    f"不支持的文档类型: {type(doc)}", code="E100"
                )
            chunks.extend(splitter.split(document))

        if not chunks:
            raise ChunkingError("未生成任何文本块，索引失败", code="E200")

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        self.vector_store.add(vectors, [c.id for c in chunks], chunks)
        self._chunks = chunks
        return len(chunks)

    def run(self, query: str, k: Optional[int] = None) -> QueryResult:
        """执行一次完整查询，返回 :class:`QueryResult`（含端到端延迟）。

        检索/生成阶段的异常按原错误码透传（``RAGNextError`` 子类）。
        """
        start = time.perf_counter()
        try:
            top_k = k if k is not None else self.config.top_k
            contexts = self.retriever.retrieve(query, top_k)
            answer = self.generator.generate(query, contexts)
        except RAGNextError:
            raise
        except Exception as exc:  # pragma: no cover - 兜底包装
            raise GenerationError(f"流水线执行失败: {exc}", code="E600") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return QueryResult(
            query=query,
            answer=answer,
            contexts=contexts,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: RAGConfig) -> "RAGPipeline":
        """依据 :class:`RAGConfig` 装配具体实现。

        默认走离线兜底链路；当配置指向 SOTA 后端（minilm / faiss / openai /
        rerank）且依赖可用时启用，否则在构造对应类时抛出对应错误码。
        """
        from ragnext.embedding.local import LocalHashingEmbedder
        from ragnext.embedding.minilm import MiniLMEmbedder
        from ragnext.generation.extractive import ExtractiveGenerator
        from ragnext.generation.llm import OpenAICompatibleGenerator
        from ragnext.retrieval.retriever import (
            CrossEncoderReranker,
            TopKRetriever,
        )
        from ragnext.vectorstore.faiss_store import FAISSVectorStore
        from ragnext.vectorstore.numpy_store import NumpyVectorStore

        # 嵌入器
        if config.embedder_type == "minilm":
            embedder: Embedder = MiniLMEmbedder(config.model_name)
        else:
            embedder = LocalHashingEmbedder(dim=config.embedder_dim)

        # 向量库
        if config.store_type == "faiss":
            vector_store: VectorStore = FAISSVectorStore(metric=config.metric)
        else:
            vector_store = NumpyVectorStore(metric=config.metric)

        # 可选重排器（依赖不可用时降级为不使用重排，保证可跑）
        reranker = None
        if config.use_rerank:
            try:
                reranker = CrossEncoderReranker(config.reranker_model)
            except RAGNextError:
                reranker = None

        retriever: Retriever = TopKRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=config.top_k,
            use_rerank=config.use_rerank,
            reranker=reranker,
        )

        # 生成器
        if config.generator_type == "openai":
            generator: Generator = OpenAICompatibleGenerator(
                model=config.openai_model,
                api_key_env=config.api_key_env,
                base_url_env=config.base_url_env,
            )
        else:
            generator = ExtractiveGenerator(
                max_sentences=config.max_answer_sentences,
                max_chars=config.max_answer_chars,
            )

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            generator=generator,
            config=config,
        )

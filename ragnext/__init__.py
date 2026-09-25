"""RAGNext — 模块化检索增强生成（RAG）系统。

对外导出核心类型、异常、配置、接口与流水线，并提供包版本号 ``__version__``。
所有 SOTA 后端（MiniLM / FAISS / OpenAI / ragas）均在各自模块内以
``importorskip`` 方式隔离，导入本包不会触发任何外部模型下载或密钥读取。
"""

from __future__ import annotations

from ragnext.core.types import (
    Chunk,
    Document,
    QueryResult,
    SearchResult,
)
from ragnext.core.errors import (
    RAGNextError,
    ChunkingError,
    EmbeddingError,
    EvalError,
    GenerationError,
    IngestionError,
    RetrievalError,
    VectorStoreError,
)
from ragnext.core.config import RAGConfig
from ragnext.core.interfaces import (
    Embedder,
    Evaluator,
    Generator,
    Retriever,
    VectorStore,
)
from ragnext.ingestion.loaders import (
    PDFLoader,
    TextLoader,
    MarkdownLoader,
    load_document,
)
from ragnext.chunking.splitter import RecursiveCharacterSplitter
from ragnext.embedding.base import BaseEmbedder
from ragnext.embedding.local import LocalHashingEmbedder
from ragnext.embedding.minilm import MiniLMEmbedder
from ragnext.vectorstore.base import VectorStore as VectorStoreBase
from ragnext.vectorstore.numpy_store import NumpyVectorStore
from ragnext.vectorstore.faiss_store import FAISSVectorStore
from ragnext.retrieval.retriever import TopKRetriever, CrossEncoderReranker
from ragnext.generation.base import Generator as GeneratorBase
from ragnext.generation.extractive import ExtractiveGenerator
from ragnext.generation.llm import OpenAICompatibleGenerator
from ragnext.eval.metrics import DeterministicEvaluator
from ragnext.pipeline.rag_pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core types
    "Document",
    "Chunk",
    "SearchResult",
    "QueryResult",
    # errors
    "RAGNextError",
    "IngestionError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "GenerationError",
    "EvalError",
    # config & interfaces
    "RAGConfig",
    "Embedder",
    "VectorStore",
    "Retriever",
    "Generator",
    "Evaluator",
    # ingestion
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "load_document",
    # chunking
    "RecursiveCharacterSplitter",
    # embedding
    "BaseEmbedder",
    "LocalHashingEmbedder",
    "MiniLMEmbedder",
    # vectorstore
    "VectorStoreBase",
    "NumpyVectorStore",
    "FAISSVectorStore",
    # retrieval
    "TopKRetriever",
    "CrossEncoderReranker",
    # generation
    "GeneratorBase",
    "ExtractiveGenerator",
    "OpenAICompatibleGenerator",
    # eval
    "DeterministicEvaluator",
    # pipeline
    "RAGPipeline",
]

# RAGNext 架构文档（Architecture）

> 面向实现者与评审者；与仓库根 `ragnext-design.md` 开工基线保持一致。

## 1. 系统定位

- 形态：Python 包 `ragnext` + CLI/示例 + 测试 + 文档 + Dockerfile。
- 核心链路：`Ingestion → Chunking → Embedding → VectorStore → Retrieval → Generation → Eval`。
- 离线兜底：`LocalHashingEmbedder` + `NumpyVectorStore` + `ExtractiveGenerator`。
- SOTA 可插拔：`MiniLMEmbedder` / `FAISSVectorStore` / `OpenAICompatibleGenerator` / `RagasEvaluator`。

## 2. 模块与职责

| 模块 | 主要类 / 函数 | 输入 | 输出 | 错误码 |
|------|---------------|------|------|--------|
| core.types | `Document` / `Chunk` / `SearchResult` / `QueryResult` | — | 数据结构 | — |
| core.errors | `RAGNextError` + `E100~E700` | — | 带码异常 | E100–E700 |
| core.interfaces | `Embedder` / `VectorStore` / `Retriever` / `Generator` / `Evaluator`（Protocol） | — | 契约 | — |
| core.config | `RAGConfig` | — | 运行配置 | — |
| ingestion.loaders | `load_document` / `TextLoader` / `MarkdownLoader` / `PDFLoader` | 路径 | `Document` | E100 |
| chunking.splitter | `RecursiveCharacterSplitter.split` | `Document` | `list[Chunk]` | E200 |
| embedding.local | `LocalHashingEmbedder.embed` | `list[str]` | `(N,dim)` | E300 |
| embedding.minilm | `MiniLMEmbedder` | `list[str]` | `(N,dim)` | E300 |
| vectorstore.numpy_store | `NumpyVectorStore.add/search/save/load` | 向量 / 查询 | `list[SearchResult]` | E400 |
| vectorstore.faiss_store | `FAISSVectorStore` | 同上 | 同上 | E400 |
| retrieval.retriever | `TopKRetriever.retrieve` / `CrossEncoderReranker` | 查询串 | `list[Chunk]` | E500 |
| generation.extractive | `ExtractiveGenerator.generate` | 查询 + 上下文 | 答案 | E600 |
| generation.llm | `OpenAICompatibleGenerator.generate` | 查询 + 上下文 | 答案 | E600 |
| eval.metrics | `recall_at_k` / `latency_ms` / `context_precision` / `DeterministicEvaluator` | 评测输入 | 指标字典 | E700 |
| eval.ragas_eval | `RagasEvaluator.evaluate` | 同上 | ragas 指标 | E700 |
| pipeline.rag_pipeline | `RAGPipeline.index/run/from_config` | 文档 / 查询 | `QueryResult` | 透传各码 |

## 3. 调用关系（严格单向，无环）

```
RAGPipeline ──▶ Ingestion ──▶ Chunking ──▶ Embedding ──▶ VectorStore
                                          │                  │
                                          └──── search ◀─────┘
                                                   │ top-k
                                                   ▼
                                              Retrieval ──(rerank)──▶ Generation ──▶ Eval
```

`RAGPipeline` 仅依赖 Protocol，不依赖具体实现；`from_config` 按 `RAGConfig` 装配实现。

## 4. 接口契约（签名）

- `Embedder.embed(texts: list[str]) -> np.ndarray  # shape (N, dim)`（含 `dim` 属性）
- `VectorStore.add(vectors, ids, payloads)` / `search(q, k) -> list[SearchResult]` / `save/load`
- `Retriever.retrieve(query: str, k: int) -> list[Chunk]`
- `Generator.generate(query: str, contexts: list[Chunk]) -> str`
- `Evaluator.evaluate(query, answer, contexts, ground_truth=None) -> dict[str, float]`

## 5. 错误码

| 码 | 含义 | 触发点 |
|----|------|--------|
| E100 | IngestionError | 文件不存在 / 类型不支持 / pypdf 缺失 |
| E200 | ChunkingError | 空文本 / 非法 chunk 参数 |
| E300 | EmbeddingError | 维度非法 / 模型缺失（MiniLM） |
| E400 | VectorStoreError | 未构建 / 维度不一致 / IO 失败 / FAISS 缺失 |
| E500 | RetrievalError | 空查询 / 无结果 / CrossEncoder 缺失 |
| E600 | GenerationError | 无上下文 / LLM 调用异常 / openai 缺失 |
| E700 | EvalError | 指标非法（空 ground truth）/ ragas 缺失 |

所有异常带 `code` 属性，流水线按原码透传，CLI 据此友好展示。

## 6. 关键选型依据

- 向量库：**FAISS**（工业级 ANN 标杆，CPU 友好，无服务依赖）；兜底用 Numpy 精确检索。
- 嵌入：**all-MiniLM-L6-v2**（体积≈80MB，许可 Apache-2.0，社区最广）；兜底用哈希嵌入。
- 生成：抽取式兜底（免密钥）+ OpenAI 兼容 API（接 vLLM/Ollama/OpenAI）。
- PDF：**pypdf**（纯 Py、轻量、无系统依赖）。
- 评测：**ragas**（RAG 事实标准）+ 自研确定性指标（保证离线可测）。
- 测试：**pytest**（fixture 友好，CI 式全绿）。

## 7. 本地兜底实现说明（自研理由）

三条兜底均为**确定性、零下载、可单测**：

1. `LocalHashingEmbedder`：用 `hashlib`（不受 `PYTHONHASHSEED` 影响）对文本做字符级二元文法
   （bigram）哈希，带符号 simhash 风格累加并 L2 归一化 → 可复现、余弦可分辨。
2. `NumpyVectorStore`：numpy 精确 cosine / l2 检索，`.npz`（`allow_pickle`）持久化任意负载。
3. `ExtractiveGenerator`：从检索块中按与查询的词元重叠度抽取最相关句子拼装答案，免 LLM。

目的：HuggingFace 网络不可达 / 无 LLM 密钥时，demo 与单测零手工干预跑通（DoD）。

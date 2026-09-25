# RAGNext 使用文档（Usage）

## 1. Python API

### 1.1 默认离线链路（最小可运行）

```python
from ragnext.core.config import RAGConfig
from ragnext.pipeline.rag_pipeline import RAGPipeline
from ragnext.core.types import Document

cfg = RAGConfig()                      # 默认 local / numpy / extractive
pipeline = RAGPipeline.from_config(cfg)

docs = [
    Document(id="faq1", text="如何重置密码：在登录页点击忘记密码并输入验证码。"),
    Document(id="faq2", text="支持微信支付与支付宝两种移动支付方式。"),
]
pipeline.index(docs)                  # 也可传文件路径字符串列表
result = pipeline.run("如何重置密码？")
print(result.answer)                   # 抽取式答案
print(result.latency_ms)               # 端到端延迟（毫秒）
for c in result.contexts:
    print(c.id, c.text)
```

### 1.2 逐模块调用（独立验证）

```python
from ragnext.ingestion.loaders import load_document
from ragnext.chunking.splitter import RecursiveCharacterSplitter
from ragnext.embedding.local import LocalHashingEmbedder
from ragnext.vectorstore.numpy_store import NumpyVectorStore

doc = load_document("./docs/faq.txt")
chunks = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=20).split(doc)
emb = LocalHashingEmbedder(dim=256)
store = NumpyVectorStore(metric="cosine")
store.add(emb.embed([c.text for c in chunks]),
          [c.id for c in chunks], chunks)
hits = store.search(emb.embed(["如何重置密码？"])[0], k=3)
```

### 1.3 切换后端（SOTA）

```python
cfg = RAGConfig(
    embedder_type="minilm",   # MiniLM（sentence-transformers）
    store_type="faiss",       # FAISS（faiss-cpu）
    generator_type="openai",  # OpenAI 兼容（vLLM/Ollama/OpenAI）
    use_rerank=True,          # CrossEncoder 重排
)
pipeline = RAGPipeline.from_config(cfg)
# 需先 export OPENAI_API_KEY / OPENAI_BASE_URL
```

### 1.4 持久化索引

```python
pipeline.vector_store.save("./idx.npz")
# 另起进程：
pipeline.vector_store.load("./idx.npz")
result = pipeline.run("如何重置密码？")
```

## 2. 命令行（CLI）

```bash
python -m ragnext.cli demo                         # 内置离线示例
python -m ragnext.cli index ./docs/faq.txt -o ./idx.npz
python -m ragnext.cli query "如何重置密码？" -i ./idx.npz
# 可选参数：--chunk-size / --top-k / --embedder / --store / --generator
```

## 3. 评测（Eval）

### 3.1 确定性指标（离线）

```python
from ragnext.eval.metrics import recall_at_k, context_precision, latency_ms

recall_at_k(retrieved_ids=["c0","c2"], relevant_ids=["c0","c1"], k=2)  # 0.5
context_precision(contexts, query)        # 与查询有词元重叠的上下文占比
latency_ms(0.5)                            # -> 500.0
```

### 3.2 ragas 指标（可选）

```python
from ragnext.eval.ragas_eval import RagasEvaluator
ev = RagasEvaluator(llm=my_langchain_llm)   # 需网络与 LLM
ev.evaluate(query, answer, contexts, ground_truth="...")
```

## 4. 基准（Benchmark）

```bash
make benchmark
# 生成 scripts/benchmark_report.json：recall@1/3/5、p50/p95 延迟，
# 以及（若可用）Local-vs-MiniLM、Numpy-vs-FAISS 对比。
```

实测基线（本机 CPU，50 条中文 FAQ）：Local+Numpy **recall@1=recall@5=1.000**，
检索 p50≈0.03ms；FAISS 同样全命中。

## 5. 常见问题

- **demo 报 E300/E400/E500/E600？** 说明某可选后端被隐式启用但依赖缺失；检查
  `RAGConfig` 是否误设为 `minilm/faiss/openai`，默认离线链路不会触发。
- **检索结果不相关？** 调小 `chunk_size` 使每个块语义更单一，或启用 `use_rerank=True`。
- **中文区分度差？** 嵌入器已采用字符级二元文法（bigram），一般无需额外处理。

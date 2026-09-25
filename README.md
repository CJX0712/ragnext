# RAGNext · 模块化检索增强生成（RAG）系统

> 一套**可实际运行、CPU 可跑、一键复现、模块可独立验证**的检索增强生成系统。
> 默认走**离线兜底链路**（零下载、零密钥即可跑通），并支持**可插拔 SOTA 后端**
> （MiniLM / FAISS / OpenAI 兼容 / ragas）。

- 版本：`__version__ == "0.1.0"`
- 许可：MIT
- 作者：晨星

---

## 1. 特性

| 能力 | 默认（离线兜底） | 可选 SOTA 后端 |
|------|------------------|----------------|
| 嵌入 Embedding | `LocalHashingEmbedder`（纯 numpy，确定性） | `MiniLMEmbedder`（sentence-transformers） |
| 向量库 VectorStore | `NumpyVectorStore`（精确检索，`.npz` 持久化） | `FAISSVectorStore`（faiss-cpu） |
| 生成 Generation | `ExtractiveGenerator`（抽取式，免 LLM） | `OpenAICompatibleGenerator`（vLLM/Ollama/OpenAI） |
| 评测 Eval | `DeterministicEvaluator`（recall@k / latency / context_precision） | `RagasEvaluator`（faithfulness / answer_relevancy） |
| 重排 Rerank | — | `CrossEncoderReranker`（sentence-transformers） |

核心链路严格单向：`Ingestion → Chunking → Embedding → VectorStore → Retrieval → Generation → Eval`，
由 `RAGPipeline` 编排；流水线只依赖**接口（Protocol）**，实现可替换、可独立单测。

---

## 2. 快速开始

### 2.1 安装（venv 隔离，失败容忍可选依赖）

```bash
make install          # 建 .venv 并安装核心依赖；可选 SOTA 依赖失败不影响主线
# 或
bash scripts/setup.sh # Linux/macOS
powershell scripts/setup.ps1  # Windows
```

### 2.2 一键离线示例

```bash
make demo
# 等价：python -m ragnext.examples.run_demo
```

输出：加载内置 FAQ → 切分 → 本地哈希嵌入 → Numpy 建库 → 检索 → 抽取式生成 →
打印答案 + 命中块 + 端到端延迟。

### 2.3 Python API（最小示例）

```python
from ragnext.core.config import RAGConfig
from ragnext.pipeline.rag_pipeline import RAGPipeline

cfg = RAGConfig(embedder_type="local", store_type="numpy", generator_type="extractive")
pipeline = RAGPipeline.from_config(cfg)
pipeline.index(["./docs/faq.txt"])          # 也支持直接传 Document
result = pipeline.run("如何重置密码？")
print(result.answer, result.latency_ms)
```

### 2.4 CLI

```bash
python -m ragnext.cli demo
python -m ragnext.cli index ./docs/faq.txt -o ./idx.npz
python -m ragnext.cli query "如何重置密码？" -i ./idx.npz
```

---

## 3. 模块结构

```
ragnext/
├── core/          # types / errors / interfaces / config（共享契约）
├── ingestion/     # loaders：Text / Markdown / PDF
├── chunking/      # RecursiveCharacterSplitter（递归切分 + overlap）
├── embedding/     # base / local（哈希兜底）/ minilm（MiniLM）
├── vectorstore/   # base / numpy_store（精确兜底）/ faiss_store
├── retrieval/     # retriever：TopKRetriever + 可选 CrossEncoderReranker
├── generation/    # base / extractive（抽取兜底）/ llm（OpenAI 兼容）
├── eval/          # metrics（确定性）/ ragas_eval（可选）
├── pipeline/      # rag_pipeline：编排 + 错误码透传
├── cli.py         # argparse 子命令 index/query/demo
└── examples/      # run_demo：内置离线端到端示例
tests/             # 10 个离线单测模块（pytest 全绿）
scripts/           # benchmark.py / setup.sh / setup.ps1
docs/              # architecture / deployment / usage
```

---

## 4. 启用 SOTA 后端（需网络 / 密钥）

```bash
pip install "ragnext[sota]"        # 或 pip install -r requirements-sota.txt
export RAGNEXT_EMBEDDER=minilm RAGNEXT_GENERATOR=openai \
       OPENAI_API_KEY=... OPENAI_BASE_URL=http://localhost:8000/v1
python -m ragnext.examples.run_demo
```

通过 `RAGConfig` 切换：`embedder_type ∈ {local,minilm}`、`store_type ∈ {numpy,faiss}`、
`generator_type ∈ {extractive,openai}`、`use_rerank=True`。任意可选依赖缺失时自动跳过，
不影响离线主线（见各模块 `importorskip` 隔离）。

> 环境变量一键切换（由 `RAGPipeline.from_config` 读取；未设置时保持显式配置/默认值）：
> `RAGNEXT_EMBEDDER`(local|minilm)、`RAGNEXT_STORE`(numpy|faiss)、
> `RAGNEXT_GENERATOR`(extractive|openai)、`RAGNEXT_MODEL`(模型名)、`RAGNEXT_RERANK`(0|1)。
> LLM 密钥 / 地址仍走 `OPENAI_API_KEY` / `OPENAI_BASE_URL`（由 `OpenAICompatibleGenerator` 读取）。

---

## 5. 测试与基准

```bash
make test            # pytest 全模块单测（离线）
make benchmark       # 50 条语料上测 recall@k 与 p50/p95 延迟，输出 benchmark_report.json
```

实测基线（本机 CPU）：Local+Numpy 在 50 条中文 FAQ 上 **recall@1 = recall@5 = 1.000**，
检索 p50 ≈ 0.03ms；FAISS 后端同样全命中（小规模语料下 Numpy 精确检索已足够快）。

---

## 6. 文档

- `docs/architecture.md`：架构、模块职责、接口契约、错误码、选型依据
- `docs/deployment.md`：本地 / Docker / 可选后端环境变量 / 离线保证
- `docs/usage.md`：Python API、CLI、配置与后端切换、评测用法

---

## 7. 设计原则

禁止从零自研核心算法：向量检索用 FAISS、稠密向量用 all-MiniLM-L6-v2、PDF 解析用 pypdf、
数值用 numpy、评测用 ragas。仅“本地兜底”三条（哈希嵌入 / Numpy 精确检索 / 抽取式生成）
为**确定性、零下载、可单测**的自研实现，目的是在网络/密钥不可达时保证系统可跑通（DoD）。

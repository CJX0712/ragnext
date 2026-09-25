# RAGNext 部署文档（Deployment）

## 1. 环境要求

- Python 3.10+（开发验证于 3.13）
- CPU 即可运行（无需 GPU）
- 离线兜底链路**无需任何网络下载或 API 密钥**

## 2. 本地部署（venv 隔离）

```bash
# 1) 创建并激活虚拟环境
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) 安装核心依赖（离线兜底链路）
pip install -r requirements.txt

# 3) 安装可选 SOTA 后端（失败不影响主线）
pip install -r requirements-sota.txt || echo "跳过可选依赖"

# 4) 可编辑安装本包（提供 CLI / 示例入口）
pip install -e .
```

或使用一键脚本：`make install` / `bash scripts/setup.sh` / `powershell scripts/setup.ps1`。

## 3. Docker 部署

```bash
docker build -t ragnext .
docker run --rm ragnext make demo
```

镜像基于 `python:3.13-slim`，默认安装核心依赖并运行离线 `make demo`；`make install` 会
在容器内建立 `.venv`。可选 SOTA 依赖（需联网）可在构建阶段自行加入：
`RUN pip install -r requirements-sota.txt`。

## 4. 启用可选 SOTA 后端

| 后端 | 依赖 | 启用方式 |
|------|------|----------|
| MiniLM 嵌入 | `sentence-transformers` | `RAGConfig(embedder_type="minilm")` |
| FAISS 向量库 | `faiss-cpu` | `RAGConfig(store_type="faiss")` |
| OpenAI 兼容生成 | `openai` | `RAGConfig(generator_type="openai")` + 环境变量 |
| CrossEncoder 重排 | `sentence-transformers` | `RAGConfig(use_rerank=True)` |
| ragas 评测 | `ragas` | 使用 `RagasEvaluator` |

LLM 相关环境变量（由 `OpenAICompatibleGenerator` 读取）：

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1      # 或 vLLM/Ollama 地址
# 自定义键名：RAGConfig(api_key_env="MY_KEY", base_url_env="MY_URL")
```

> 任意可选依赖缺失时，对应模块以 `importorskip` 隔离：导入安全、构造时抛出对应错误码
> （E300/E400/E500/E600/E700），且 `RAGPipeline.from_config` 在重排不可用时自动降级为不重排。

## 5. 离线保证（DoD）

- 不安装任何可选依赖时：`LocalHashingEmbedder` + `NumpyVectorStore` + `ExtractiveGenerator`
  即可完成「加载 → 切分 → 嵌入 → 建库 → 检索 → 生成」。
- `make demo` / `make test` / `make benchmark` 全部离线可跑。
- 依赖版本全部 pin 死（`requirements.txt` / `requirements-sota.txt`），并用 `requirements.lock.txt`
  （`pip freeze`）锁定完整传递闭包，保证可复现。

## 6. 持久化与运维

- 向量库：`NumpyVectorStore.save(path)` 落 `.npz`；`FAISSVectorStore.save(path)` 落索引文件 +
  `.meta.npz`（ids/payloads）。
- 加载：CLI `query` 子命令或 `RAGPipeline.vector_store.load(path)` 复用已建索引，避免重复嵌入。
- 检索延迟在 `QueryResult.latency_ms` 中产出，便于线上监控与基准对比。

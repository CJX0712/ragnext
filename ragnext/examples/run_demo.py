"""内置离线示例：加载 → 切分 → 本地嵌入 → Numpy 建库 → 检索 → 抽取式生成。

该脚本**零下载、零密钥**即可端到端跑通，用于验证默认兜底链路：
    LocalHashingEmbedder + NumpyVectorStore + ExtractiveGenerator

运行方式：
    python -m ragnext.examples.run_demo
"""

from __future__ import annotations

import os

from ragnext.core.config import RAGConfig
from ragnext.ingestion.loaders import load_document
from ragnext.pipeline.rag_pipeline import RAGPipeline

# 演示查询（与 sample_faq.txt 中的主题对应）。
DEMO_QUERIES = [
    "如何重置密码？",
    "支持哪些支付方式？",
    "怎么申请退款？",
    "会员有什么权益？",
    "优惠券怎么用？",
]

SEPARATOR = "=" * 60


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    faq_path = os.path.join(here, "sample_faq.txt")

    # 1) 加载
    doc = load_document(faq_path)
    print(f"{SEPARATOR}\n[1] 加载文档：{doc.metadata.get('source')}（{len(doc.text)} 字）")

    # 2) 配置（默认离线兜底链路）
    config = RAGConfig(
        chunk_size=160,
        chunk_overlap=20,
        top_k=3,
        embedder_type="local",
        store_type="numpy",
        generator_type="extractive",
        embedder_dim=256,
    )

    # 3) 构建流水线并索引
    pipeline = RAGPipeline.from_config(config)
    n_chunks = pipeline.index([doc])
    print(f"[2] 切分并索引：{n_chunks} 个文本块 "
          f"（embedder=local dim={config.embedder_dim}, store=numpy）")

    # 4) 逐条查询：检索 + 抽取式生成
    print(f"{SEPARATOR}\n[3] 离线问答演示\n{SEPARATOR}")
    for query in DEMO_QUERIES:
        result = pipeline.run(query)
        print(f"\n❓ 查询：{result.query}")
        print(f"💡 答案：{result.answer}")
        print(f"⏱  延迟：{result.latency_ms:.2f} ms ｜ 命中块：{len(result.contexts)}")
        for i, ctx in enumerate(result.contexts, 1):
            snippet = ctx.text.replace("\n", " ")
            print(f"   · 块{i} [{ctx.id}] {snippet[:60]}...")
    print(f"\n{SEPARATOR}\n[demo] 完成 ✅ 离线兜底链路运行正常。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

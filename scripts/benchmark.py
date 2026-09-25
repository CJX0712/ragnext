"""RAGNext 基准测试脚本（离线优先，可选后端对标）。

默认使用离线兜底链路（LocalHashingEmbedder + NumpyVectorStore）在 50 条
中文 FAQ 语料上测量：
    * recall@1 / recall@3 / recall@5
    * 检索延迟 p50 / p95（毫秒）

若环境中可用，则额外对标：
    * Local vs MiniLM 嵌入（recall + 延迟）
    * Numpy vs FAISS 向量库（延迟）

结果同时打印到终端并写入 ``benchmark_report.json``。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 允许以脚本方式直接运行（python scripts/benchmark.py）时也能导入 ragnext 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ragnext.chunking.splitter import RecursiveCharacterSplitter
from ragnext.core.types import Document
from ragnext.embedding.local import LocalHashingEmbedder
from ragnext.eval.metrics import recall_at_k
from ragnext.retrieval.retriever import TopKRetriever
from ragnext.vectorstore.numpy_store import NumpyVectorStore

# 50 个中文主题，用于程序化生成 FAQ 语料（确定性、离线）。
TOPICS = [
    "密码重置", "支付方式", "订单发货", "申请退款", "账号注销",
    "会员权益", "优惠券使用", "人工客服", "隐私保护", "数据加密",
    "发票开具", "物流跟踪", "商品退换", "价格保护", "积分规则",
    "签到任务", "实名认证", "绑定手机", "修改邮箱", "夜间模式",
    "消息通知", "收藏功能", "搜索历史", "分享好友", "青少年模式",
    "企业认证", "批量导出", "API 调用", "限流策略", "错误码含义",
    "版本更新", "缓存清理", "离线模式", "多端同步", "语音输入",
    "图片识别", "自动续费", "试用期限", "发票抬头", "跨境运费",
    "关税计算", "质保期限", "维修流程", "以旧换新", "预约服务",
    "门店查询", "营业时间", "投诉建议", "安全中心", "账号申诉",
]


def _build_corpus(n: int = 50) -> List[Document]:
    docs: List[Document] = []
    for i, topic in enumerate(TOPICS[:n]):
        question = f"如何办理{topic}？"
        answer = (
            f"{topic}请在「账户—{topic}」页面操作，"
            f"处理时效约 {(i % 9) + 1} 个工作日，"
            f"结果将以站内信通知。详情参见帮助中心的{topic}专题。"
        )
        docs.append(
            Document(
                id=f"doc{i}",
                text=f"问：{question}\n答：{answer}",
                metadata={"topic": topic, "question": question},
            )
        )
    return docs


def _run_backend(
    docs: List[Document],
    embedder: Any,
    store_cls: Any,
    store_kwargs: Dict[str, Any],
    top_k: int,
    k_eval: int,
) -> Dict[str, float]:
    splitter = RecursiveCharacterSplitter(chunk_size=300, chunk_overlap=20)
    chunks: List[Any] = []
    relevant: Dict[str, List[str]] = {}
    for d in docs:
        cs = splitter.split(d)
        relevant[d.id] = [c.id for c in cs]
        chunks.extend(cs)

    store = store_cls(**store_kwargs)
    vectors = embedder.embed([c.text for c in chunks])
    store.add(vectors, [c.id for c in chunks], chunks)
    retriever = TopKRetriever(embedder, store, top_k=top_k)

    latencies: List[float] = []
    recalls = {"k1": [], "k3": [], "k5": []}
    for d in docs:
        q = d.metadata["question"]
        t0 = time.perf_counter()
        got = retriever.retrieve(q, k=k_eval)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        got_ids = [c.id for c in got]
        recalls["k1"].append(recall_at_k(got_ids, relevant[d.id], k=1))
        recalls["k3"].append(recall_at_k(got_ids, relevant[d.id], k=3))
        recalls["k5"].append(recall_at_k(got_ids, relevant[d.id], k=5))

    def pct(arr: List[float], p: float) -> float:
        return float(np.percentile(np.asarray(arr), p))

    return {
        "recall@1": float(np.mean(recalls["k1"])),
        "recall@3": float(np.mean(recalls["k3"])),
        "recall@5": float(np.mean(recalls["k5"])),
        "latency_p50_ms": pct(latencies, 50),
        "latency_p95_ms": pct(latencies, 95),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
    }


def main() -> int:
    n_docs = 50
    top_k = 5
    k_eval = max(top_k, 5)
    docs = _build_corpus(n_docs)
    print(f"[benchmark] 语料：{n_docs} 条 FAQ")

    report: Dict[str, Any] = {}

    # 1) 离线兜底基线：Local + Numpy
    local = LocalHashingEmbedder(dim=256)
    baseline = _run_backend(docs, local, NumpyVectorStore, {"metric": "cosine"}, top_k, k_eval)
    report["baseline_local_numpy"] = baseline
    print(
        f"[baseline] Local+Numpy  recall@1={baseline['recall@1']:.3f} "
        f"recall@5={baseline['recall@5']:.3f}  "
        f"p50={baseline['latency_p50_ms']:.3f}ms p95={baseline['latency_p95_ms']:.3f}ms"
    )

    # 2) 可选对标：Numpy vs FAISS（同 Local 嵌入，仅换向量库）
    try:
        from ragnext.vectorstore.faiss_store import FAISSVectorStore

        faiss_res = _run_backend(
            docs, local, FAISSVectorStore, {"metric": "cosine"}, top_k, k_eval
        )
        report["compare_numpy_vs_faiss"] = {
            "numpy": baseline,
            "faiss": faiss_res,
            "latency_speedup_p50": baseline["latency_p50_ms"] / max(faiss_res["latency_p50_ms"], 1e-9),
        }
        print(
            f"[compare]   Local+FAISS p50={faiss_res['latency_p50_ms']:.3f}ms "
            f"(speedup {report['compare_numpy_vs_faiss']['latency_speedup_p50']:.2f}x)"
        )
    except Exception as exc:  # pragma: no cover - FAISS 不可用
        report["compare_numpy_vs_faiss"] = {"error": str(exc)}

    # 3) 可选对标：Local vs MiniLM（同 Numpy 库，仅换嵌入器）
    try:
        from ragnext.embedding.minilm import MiniLMEmbedder

        minilm = MiniLMEmbedder("all-MiniLM-L6-v2")
        minilm_res = _run_backend(
            docs, minilm, NumpyVectorStore, {"metric": "cosine"}, top_k, k_eval
        )
        report["compare_local_vs_minilm"] = {
            "local": baseline,
            "minilm": minilm_res,
        }
        print(
            f"[compare]   MiniLM+Numpy recall@1={minilm_res['recall@1']:.3f} "
            f"p50={minilm_res['latency_p50_ms']:.3f}ms"
        )
    except Exception as exc:  # pragma: no cover - MiniLM 不可用
        report["compare_local_vs_minilm"] = {"error": str(exc)}

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[benchmark] 报告已写入 {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

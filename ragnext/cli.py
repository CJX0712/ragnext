"""RAGNext 命令行入口。

子命令：
    index   索引一个或多个文档，保存向量库到磁盘。
    query   从磁盘载入索引并对单条查询返回答案。
    demo    运行内置离线示例（等价于 ``python -m ragnext.examples.run_demo``）。

示例：
    python -m ragnext.cli demo
    python -m ragnext.cli index ./docs/faq.txt -o ./index.npz
    python -m ragnext.cli query "如何重置密码？" -i ./index.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from ragnext.core.config import RAGConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragnext",
        description="RAGNext 模块化检索增强生成系统（CLI）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="索引文档并保存向量库")
    p_index.add_argument("paths", nargs="+", help="待索引的文件路径")
    p_index.add_argument("-o", "--output", default="./ragnext.index.npz", help="索引输出路径")
    p_index.add_argument("--chunk-size", type=int, default=RAGConfig.chunk_size)
    p_index.add_argument("--chunk-overlap", type=int, default=RAGConfig.chunk_overlap)
    p_index.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    p_index.add_argument("--embedder", default=RAGConfig.embedder_type, choices=["local", "minilm"])
    p_index.add_argument("--store", default=RAGConfig.store_type, choices=["numpy", "faiss"])
    p_index.add_argument("--generator", default=RAGConfig.generator_type, choices=["extractive", "openai"])

    p_query = sub.add_parser("query", help="对查询返回答案")
    p_query.add_argument("query", help="查询文本")
    p_query.add_argument("-i", "--index", default="./ragnext.index.npz", help="索引路径")
    p_query.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    p_query.add_argument("--embedder", default=RAGConfig.embedder_type, choices=["local", "minilm"])
    p_query.add_argument("--store", default=RAGConfig.store_type, choices=["numpy", "faiss"])
    p_query.add_argument("--generator", default=RAGConfig.generator_type, choices=["extractive", "openai"])

    sub.add_parser("demo", help="运行内置离线示例")
    return parser


def cmd_index(args: argparse.Namespace) -> int:
    from ragnext.pipeline.rag_pipeline import RAGPipeline

    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedder_type=args.embedder,
        store_type=args.store,
        generator_type=args.generator,
    )
    pipeline = RAGPipeline.from_config(config)
    n = pipeline.index(list(args.paths))
    pipeline.vector_store.save(args.output)
    print(f"[index] 已索引 {n} 个文本块 → {args.output}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from ragnext.pipeline.rag_pipeline import RAGPipeline

    config = RAGConfig(
        top_k=args.top_k,
        embedder_type=args.embedder,
        store_type=args.store,
        generator_type=args.generator,
    )
    pipeline = RAGPipeline.from_config(config)
    if not os.path.exists(args.index):
        print(f"[query] 索引文件不存在: {args.index}", file=sys.stderr)
        return 1
    pipeline.vector_store.load(args.index)
    result = pipeline.run(args.query)
    print(f"\n[query] {result.query}")
    print(f"[answer] {result.answer}")
    print(f"[latency] {result.latency_ms:.2f} ms，命中 {len(result.contexts)} 块")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from ragnext.examples.run_demo import main as demo_main

    return demo_main()


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "demo":
        return cmd_demo(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

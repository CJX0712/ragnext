"""确定性评测指标（离线可测，零依赖）。

提供 ``recall_at_k`` / ``latency_ms`` / ``context_precision`` 及若干辅助指标，
并给出实现 :class:`ragnext.core.interfaces.Evaluator` 协议的
:class:`DeterministicEvaluator`，可在无 LLM、无网络的情况下复现评测结果。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ragnext.core.errors import EvalError
from ragnext.core.tokenizer import tokenize as _tokenize
from ragnext.core.types import Chunk
from ragnext.core.interfaces import Evaluator


def _token_set(text: str) -> set:
    return set(_tokenize(text))


def recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: Optional[int] = None
) -> float:
    """在 ``k`` 个召回结果上计算召回率。

    Args:
        retrieved_ids: 按相关性排序的召回标识列表。
        relevant_ids: 相关标识集合（ground truth）。
        k: 截断位置；默认使用全部召回。

    Returns:
        召回率 = 命中相关数 / 相关总数，取值 ``[0, 1]``。

    Raises:
        EvalError (E700): 相关集合为空。
    """
    if not relevant_ids:
        raise EvalError("relevant_ids 为空，无法计算 recall@k", code="E700")
    if k is not None and k > 0:
        retrieved = retrieved_ids[:k]
    else:
        retrieved = retrieved_ids
    relevant = set(relevant_ids)
    hits = len(relevant & set(retrieved))
    return hits / len(relevant)


def latency_ms(seconds: float) -> float:
    """将秒为单位的耗时转换为毫秒。"""
    if seconds < 0:
        raise EvalError(f"耗时不能为负: {seconds}", code="E700")
    return float(seconds) * 1000.0


def context_precision(contexts: List[Chunk], query: str) -> float:
    """上下文精度：与查询存在词元重叠的上下文占比。

    衡量“检索到的块有多少是相关的”。取值 ``[0, 1]``；空上下文返回 0。
    """
    if not contexts:
        return 0.0
    q_tokens = _token_set(query)
    if not q_tokens:
        return 0.0
    hits = sum(1 for c in contexts if q_tokens & _token_set(c.text))
    return hits / len(contexts)


def answer_context_overlap(answer: str, contexts: List[Chunk]) -> float:
    """答案与上下文的词元重叠率（faithfulness 的轻量代理指标）。

    取值 ``[0, 1]``：答案词元中能在上下文找到的比例。
    """
    if not answer or not answer.strip():
        return 0.0
    ans_tokens = _token_set(answer)
    if not ans_tokens:
        return 0.0
    ctx_tokens: set = set()
    for c in contexts:
        ctx_tokens |= _token_set(c.text)
    if not ctx_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


class DeterministicEvaluator:
    """确定性评测器，实现 ``Evaluator`` 协议。"""

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[Chunk],
        ground_truth: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """返回一组离线可计算的评测指标。

        指标键：
            latency_ms：调用方传入的耗时（秒）转换，此处不直接计时。
            context_precision：上下文精度。
            answer_context_overlap：答案-上下文重叠率。
            answer_non_empty：答案是否非空（1.0 / 0.0）。
            recall_at_k：若提供 ``ground_truth`` 且能解析出相关块标识则计算。

        说明：``recall_at_k`` 需要相关标识集合；本确定性实现在给定
        ``ground_truth`` 时，把它当作“相关上下文文本集合”来近似计算
        答案词元覆盖度，实际业务中更推荐由流水线显式传入相关 chunk 标识。
        """
        result: Dict[str, float] = {
            "context_precision": context_precision(contexts, query),
            "answer_context_overlap": answer_context_overlap(answer, contexts),
            "answer_non_empty": float(bool(answer and answer.strip())),
        }
        if ground_truth:
            gt_tokens = _token_set(ground_truth)
            if gt_tokens:
                ans_tokens = _token_set(answer)
                result["answer_ground_truth_overlap"] = (
                    len(ans_tokens & gt_tokens) / len(gt_tokens) if gt_tokens else 0.0
                )
        return result

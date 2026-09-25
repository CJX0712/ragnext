"""eval.metrics 确定性指标测试。"""

import pytest

from ragnext.core.errors import EvalError
from ragnext.core.types import Chunk
from ragnext.eval.metrics import (
    DeterministicEvaluator,
    answer_context_overlap,
    context_precision,
    latency_ms,
    recall_at_k,
)


def test_recall_at_k_full():
    assert recall_at_k(["a", "b", "c"], ["a", "c"]) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x"], ["a", "c"]) == 0.5


def test_recall_at_k_truncated():
    assert recall_at_k(["a", "b"], ["a", "c"], k=1) == 0.5


def test_recall_empty_relevant_raises():
    with pytest.raises(EvalError):
        recall_at_k(["a"], [])


def test_latency_ms():
    assert latency_ms(0.5) == 500.0


def test_context_precision():
    ctx = [
        Chunk(id="1", doc_id="d", text="如何重置密码点击忘记密码"),
        Chunk(id="2", doc_id="d", text="今天的天气真不错"),
    ]
    assert context_precision(ctx, "如何重置密码") == 0.5


def test_answer_context_overlap():
    ctx = [Chunk(id="1", doc_id="d", text="在登录页点击忘记密码输入验证码")]
    assert answer_context_overlap("点击忘记密码", ctx) == 1.0


def test_deterministic_evaluator():
    ctx = [Chunk(id="1", doc_id="d", text="如何重置密码点击忘记密码输入验证码")]
    ev = DeterministicEvaluator()
    m = ev.evaluate("如何重置密码", "点击忘记密码", ctx)
    assert "context_precision" in m
    assert m["answer_non_empty"] == 1.0

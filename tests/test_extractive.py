"""generation.extractive 抽取式生成器测试。"""

import pytest

from ragnext.core.errors import GenerationError
from ragnext.core.types import Chunk
from ragnext.generation.extractive import ExtractiveGenerator


def _chunks():
    return [
        Chunk(id="c1", doc_id="d", text="如何重置密码：在登录页点击忘记密码并输入验证码。"),
        Chunk(id="c2", doc_id="d", text="支持微信支付与支付宝两种移动支付方式。"),
    ]


def test_generate_overlap():
    gen = ExtractiveGenerator()
    ans = gen.generate("如何重置密码", _chunks())
    assert "密码" in ans or "验证码" in ans


def test_generate_empty_contexts():
    gen = ExtractiveGenerator()
    with pytest.raises(GenerationError):
        gen.generate("q", [])


def test_generate_no_overlap_fallback():
    gen = ExtractiveGenerator()
    ans = gen.generate("量子计算是什么原理", _chunks())
    assert ans  # 兜底仍返回非空答案


def test_max_sentences_and_chars():
    gen = ExtractiveGenerator(max_sentences=1, max_chars=20)
    chunks = [
        Chunk(id="c", doc_id="d", text="第一句内容与密码相关。第二句也有关联但较长一些用于测试截断效果。")
    ]
    ans = gen.generate("密码", chunks)
    assert len(ans) <= 20

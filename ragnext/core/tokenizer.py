"""共享中文/英文分词工具（字符级二元文法 bigram）。

为保证嵌入、抽取式生成、评测指标三处对“词元重叠”的判定一致，统一在此
实现分词：连续 CJK 串展开为相邻汉字二元组，拉丁/数字词整体保留。详见
``embedding/local.py`` 中的说明。
"""

from __future__ import annotations

import re

# 中英文 + 数字词元正则（覆盖 CJK 基本区）。
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

_CJK_MIN = ord("\u4e00")
_CJK_MAX = ord("\u9fff")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return _CJK_MIN <= code <= _CJK_MAX


def tokenize(text: str) -> list:
    """将文本拆为小写特征词元（CJK 用二元文法，英文/数字用整词）。"""
    grams: list = []
    for word in _TOKEN_RE.findall((text or "").lower()):
        if all(_is_cjk(ch) for ch in word):
            if len(word) == 1:
                grams.append(word)
            else:
                grams.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            grams.append(word)
    return grams

"""本地哈希嵌入器（离线兜底，零下载、确定性）。

设计理由（见架构文档第 2 节）：当 HuggingFace 网络不可达时，仍需保证 demo
与单测零依赖跑通。本实现用 ``hashlib``（而非 Python 内置 ``hash``，后者受
``PYTHONHASHSEED`` 影响）对文本 token 做稳定哈希，采用带符号 simhash 风格
累加得到稠密向量并按 L2 归一化，使余弦相似度具备区分度。

特性：
    * 确定性：相同输入恒得相同向量，跨进程一致。
    * 可配置维度：``dim`` 默认 256，可在构造时指定。
    * 纯 numpy：无外部模型、无网络。
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from ragnext.core.tokenizer import tokenize as _tokenize
from ragnext.embedding.base import BaseEmbedder
from ragnext.core.errors import EmbeddingError


class LocalHashingEmbedder(BaseEmbedder):
    """基于稳定哈希的本地嵌入器。

    Args:
        dim: 输出向量维度，默认 256。
        normalize: 是否对输出向量做 L2 归一化（默认 ``True``，便于余弦比较）。
    """

    def __init__(self, dim: int = 256, normalize: bool = True) -> None:
        if dim <= 0:
            raise EmbeddingError(f"dim 必须为正数: {dim}", code="E300")
        self.dim = int(dim)
        self.normalize = bool(normalize)

    def _token_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big")
            bucket = value % self.dim
            # 带符号累加：用第 9 字节的最低位决定正负，提升余弦区分度。
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vec[bucket] += sign
        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        """编码文本列表为 ``(N, dim)`` 矩阵；空列表返回 ``(0, dim)``。"""
        if not isinstance(texts, (list, tuple)):
            raise EmbeddingError("texts 必须为序列", code="E300")
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        matrix = np.stack([self._token_vector(str(t)) for t in texts], axis=0)
        return matrix

"""嵌入器基类，兼容 :class:`ragnext.core.interfaces.Embedder` Protocol。"""

from __future__ import annotations

from typing import List

import numpy as np


class BaseEmbedder:
    """所有嵌入器的抽象基类。

    子类必须设置 ``dim`` 并实现 :meth:`embed`。

    Attributes:
        dim: 输出向量维度。
    """

    dim: int = 0

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为 ``(N, dim)`` 的向量矩阵。"""
        raise NotImplementedError

"""ragas 评测（可选 SOTA 后端）。

封装 ``ragas`` 计算 ``faithfulness`` / ``answer_relevancy`` 等 LLM 依赖型指标。
ragas 需要 LLM 后端与网络，因此以 ``importorskip`` 隔离：模块可被安全导入，
但仅当 ``ragas`` 安装、且提供了可用 LLM 时才能构造实例并运行，否则抛出
``E700``（:class:`EvalError`）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragnext.core.errors import EvalError
from ragnext.core.types import Chunk
from ragnext.core.interfaces import Evaluator

try:  # pragma: no cover - 取决于可选依赖是否安装
    from ragas import evaluate as _ragas_evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    _RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ragas_evaluate = None  # type: ignore[assignment]
    answer_relevancy = None  # type: ignore[assignment]
    faithfulness = None  # type: ignore[assignment]
    _RAGAS_AVAILABLE = False


class RagasEvaluator:
    """基于 ragas 的评测器。

    Args:
        llm: 可选，ragas 兼容的 langchain LLM 实例；留空则交由 ragas 使用
            其默认（需环境变量中配置 API Key 等）。

    Raises:
        EvalError (E700): ``ragas`` 未安装。
    """

    def __init__(self, llm: Any = None) -> None:
        if not _RAGAS_AVAILABLE:
            raise EvalError(
                "ragas 未安装，无法使用 ragas 评测：pip install ragas",
                code="E700",
            )
        self._llm = llm

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[Chunk],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """运行 ragas 评测，返回 ``faithfulness`` / ``answer_relevancy`` 等指标。

        Raises:
            EvalError (E700): ragas 运行异常（多为缺少 LLM / 网络）。
        """
        if not _RAGAS_AVAILABLE or _ragas_evaluate is None:
            raise EvalError("ragas 未安装", code="E700")
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - datasets 为 ragas 依赖
            raise EvalError("datasets 未安装（ragas 依赖）", code="E700") from exc

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [[c.text for c in contexts]],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy]
        try:
            result = _ragas_evaluate(dataset, metrics=metrics, llm=self._llm)
        except Exception as exc:  # pragma: no cover - LLM/网络异常
            raise EvalError(f"ragas 评测失败: {exc}", code="E700") from exc

        if hasattr(result, "to_dict"):
            return {k: float(v) for k, v in result.to_dict().items()}
        return {k: float(v) for k, v in dict(result).items()}

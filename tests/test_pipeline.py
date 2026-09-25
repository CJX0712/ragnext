"""pipeline.rag_pipeline 流水线编排测试（离线链路）。"""

from ragnext.core.config import RAGConfig
from ragnext.core.types import Document, QueryResult
from ragnext.pipeline.rag_pipeline import RAGPipeline


def _config(**overrides):
    base = dict(
        chunk_size=80,
        chunk_overlap=8,
        top_k=2,
        embedder_type="local",
        store_type="numpy",
        generator_type="extractive",
        embedder_dim=64,
    )
    base.update(overrides)
    return RAGConfig(**base)


def test_pipeline_offline():
    cfg = _config()
    pipeline = RAGPipeline.from_config(cfg)
    n = pipeline.index(
        [Document(id="faq", text="如何重置密码：点击忘记密码输入验证码。支持微信支付与支付宝。")]
    )
    assert n >= 1
    res = pipeline.run("如何重置密码")
    assert isinstance(res, QueryResult)
    assert res.answer
    assert res.latency_ms >= 0
    assert any("密码" in c.text for c in res.contexts)


def test_pipeline_save_load(tmp_path):
    cfg = _config()
    pipeline = RAGPipeline.from_config(cfg)
    pipeline.index(
        [Document(id="faq", text="如何重置密码：点击忘记密码。支持微信支付。")]
    )
    out = tmp_path / "idx.npz"
    pipeline.vector_store.save(str(out))

    pipeline2 = RAGPipeline.from_config(cfg)
    pipeline2.vector_store.load(str(out))
    res = pipeline2.run("如何重置密码")
    assert res.answer


def test_pipeline_empty_document_raises():
    cfg = _config()
    pipeline = RAGPipeline.from_config(cfg)
    import pytest

    with pytest.raises(Exception):
        pipeline.index([Document(id="empty", text="   ")])

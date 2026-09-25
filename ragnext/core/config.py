"""RAGNext 运行配置。

``RAGConfig`` 以 dataclass 形式集中管理所有可调参数，``RAGPipeline.from_config``
据此装配具体实现。默认值全部指向**离线兜底链路**，保证零下载、零密钥即可运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """RAG 系统运行配置。

    Attributes:
        chunk_size: 文本块目标长度（字符数）。
        chunk_overlap: 相邻块重叠长度（字符数），须小于 ``chunk_size``。
        top_k: 检索返回的块数量。
        use_rerank: 是否启用（可选的）CrossEncoder 重排。
        embedder_type: ``"local"``（哈希兜底）或 ``"minilm"``（sentence-transformers）。
        store_type: ``"numpy"``（精确兜底）或 ``"faiss"``（FAISS ANN）。
        generator_type: ``"extractive"``（抽取式兜底）或 ``"openai"``（兼容 API）。
        model_name: 稠密模型 / LLM 名称（ MiniLM 或 OpenAI 系列）。
        embedder_dim: 本地哈希嵌入维度（仅 ``embedder_type="local"`` 生效）。
        metric: 向量相似度度量，``"cosine"`` 或 ``"l2"``。
        reranker_model: CrossEncoder 重排模型名称。
        max_answer_sentences: 抽取式生成最多拼装的句子数。
        max_answer_chars: 抽取式生成答案字符上限。
        openai_model: OpenAI 兼容生成所用模型名。
        api_key_env: 读取 API Key 的环境变量名。
        base_url_env: 读取 API Base URL 的环境变量名。
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    use_rerank: bool = False
    embedder_type: str = "local"
    store_type: str = "numpy"
    generator_type: str = "extractive"
    model_name: str = "all-MiniLM-L6-v2"
    embedder_dim: int = 256
    metric: str = "cosine"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_answer_sentences: int = 3
    max_answer_chars: int = 300
    openai_model: str = "gpt-3.5-turbo"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"

    def as_dict(self) -> dict:
        """返回配置的可序列化字典副本。"""
        return dict(self.__dict__)

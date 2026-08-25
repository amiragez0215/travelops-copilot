from __future__ import annotations

from langchain_core.embeddings import Embeddings


def create_default_embeddings(
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> Embeddings:
    """
    创建 Chroma 文档入库和在线 Query 共用的本地 Embeddings。

    Args:
        model_name:
            Hugging Face 模型 ID 或本地模型目录。
            为 None 时读取 settings.rag_embedding_model。

        device:
            cpu、cuda 或其他 sentence-transformers 支持的设备。
            为 None 时读取 settings.rag_embedding_device。

    关键规则：
        同一个 Chroma collection 的文档和 Query 必须使用同一 Embedding 模型。
        更换模型或向量维度时，应重建 collection 或使用新的 collection 名称。
    """

    # 1. 在函数调用时读取 Settings，确保项目根目录 .env 已经生效。
    from app.common.config import settings

    resolved_model = model_name or settings.rag_embedding_model
    resolved_device = device or settings.rag_embedding_device

    # 2. 延迟导入大模型依赖；不使用 RAG 的单元测试无需加载 sentence-transformers。
    from langchain_huggingface import HuggingFaceEmbeddings

    # 3. normalize_embeddings=True 让向量适合使用余弦/内积相似度。
    return HuggingFaceEmbeddings(
        model_name=resolved_model,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": True},
    )

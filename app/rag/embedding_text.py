from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


EMBEDDING_TEXT_VERSION = "v2_metadata_enriched"


def build_embedding_text(document: Document) -> str:
    """Build the text encoded by dense retrieval without changing cited content.

    BM25 already indexes selected metadata together with the body.  Dense
    retrieval must receive the same human-meaningful identifiers (especially
    title, section, and tags), otherwise a query such as ``展览预约`` is
    needlessly compared only with an unlabelled body paragraph.  Chroma still
    stores ``page_content`` as the display document, so proposal citations and
    UI output remain the original source text.
    """

    metadata = document.metadata
    fields = (
        ("标题", metadata.get("title")),
        ("章节", metadata.get("section")),
        ("城市", metadata.get("city")),
        ("文档类型", metadata.get("doc_type")),
        ("主题", metadata.get("tags")),
        ("场景", metadata.get("scenario")),
        ("风险类型", metadata.get("risk_type")),
        ("活动类型", metadata.get("activity_types")),
    )
    header: list[str] = []
    for label, value in fields:
        text = _to_text(value)
        if text:
            header.append(f"{label}：{text}")
    return "\n".join([*header, "正文：", document.page_content.strip()]).strip()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()

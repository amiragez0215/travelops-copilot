from __future__ import annotations

from app.rag.document_loader import deduplicate_documents, load_rag_documents


def _write_doc(path, doc_id, title, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
doc_id: {doc_id}
title: {title}
doc_type: guide
city: 成都
scenario:
  - general
---

# {title}

## 正文

{body}
""",
        encoding="utf-8",
    )


def test_document_loader_adds_sha256_fingerprints(tmp_path):
    """
    Loader 应为每个 chunk 添加 document_hash 和 chunk_hash。
    """

    _write_doc(
        tmp_path / "guides" / "guide_a.md",
        "guide_a",
        "成都攻略A",
        "成都适合安排美食、茶馆和城市公园。",
    )

    documents = load_rag_documents(root_dir=tmp_path, chunk_size=200, chunk_overlap=20)

    assert documents
    metadata = documents[0].metadata

    assert len(metadata["document_hash"]) == 64
    assert len(metadata["chunk_hash"]) == 64
    assert metadata["content_hash"] == metadata["chunk_hash"]


def test_deduplicate_documents_keeps_one_exact_duplicate_source(tmp_path):
    """
    两个文件正文和语义 metadata 相同，即使文件名/doc_id 不同，也只保留一个文档。
    """

    body = "成都适合安排美食、茶馆和城市公园。"

    _write_doc(
        tmp_path / "guides" / "guide_a.md",
        "guide_a",
        "成都攻略A",
        body,
    )
    _write_doc(
        tmp_path / "guides" / "guide_b.md",
        "guide_b",
        "成都攻略B",
        body,
    )

    loaded = load_rag_documents(root_dir=tmp_path, chunk_size=200, chunk_overlap=20)
    result = deduplicate_documents(loaded)

    assert len(result.duplicates) == 1
    assert len({doc.metadata["source"] for doc in result.documents}) == 1
    assert result.duplicates[0].document_hash

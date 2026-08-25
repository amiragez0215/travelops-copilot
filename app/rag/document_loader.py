from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_DOCS_DIR = PROJECT_ROOT / "data" / "rag_docs"


# 文件夹名称到标准 doc_type 的映射。
DIRECTORY_DOC_TYPE_MAP = {
    "guides": "guide",
    "hotel_reviews": "hotel_reviews",
    "safety_notices": "safety_notice",
    "packing_checklists": "packing_checklist",
}


# 文件名和正文中常见的城市英文/中文别名。
# frontmatter 中显式填写的 city 优先级最高。
CITY_ALIASES = {
    "chengdu": "成都",
    "成都": "成都",
    "hangzhou": "杭州",
    "杭州": "杭州",
    "chongqing": "重庆",
    "重庆": "重庆",
    "shanghai": "上海",
    "上海": "上海",
    "beijing": "北京",
    "北京": "北京",
}


# 这些 metadata 会参与 document_hash。
# source、doc_id、title 等展示性字段不参与，这样可以识别“换了文件名但内容和语义完全重复”的文档。
SEMANTIC_METADATA_KEYS = (
    "doc_type",
    "city",
    "hotel_id",
    "scenario",
    "risk_type",
    "risk_level",
    "activity_types",
    "suitable_weather",
    "indoor_outdoor",
    "trip_days",
    "language",
)


@dataclass(frozen=True)
class DuplicateDocument:
    """
    一条文档级重复记录。

    kept_source:
        被保留并参与 BM25 / Chroma 的源文件。

    duplicate_source:
        内容和语义 metadata 与 kept_source 完全相同，因此被跳过的源文件。

    document_hash:
        使用 SHA-256 计算出的文档指纹。
    """

    kept_source: str
    duplicate_source: str
    document_hash: str


@dataclass(frozen=True)
class DeduplicationResult:
    """
    文档去重结果。

    documents:
        去重后真正用于 BM25 和 Chroma 的 chunks。

    duplicates:
        被识别为重复的源文件记录。
    """

    documents: list[Document]
    duplicates: list[DuplicateDocument]


def load_rag_documents(
    root_dir: str | Path | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    加载并切分 data/rag_docs 下的 Markdown 文档。

    Args:
        root_dir:
            RAG 文档根目录。默认使用 data/rag_docs。

        chunk_size:
            每个 chunk 的最大字符数。

        chunk_overlap:
            相邻 chunk 的重叠字符数。

    Returns:
        list[Document]:
            未执行文档级去重的 LangChain Document 列表。

    每个 Document 至少包含：
        - chunk_id
        - doc_id
        - source
        - section
        - doc_type
        - document_hash
        - chunk_hash
        - content_hash（兼容旧字段，值与 chunk_hash 相同）
    """

    # 1. 确定 RAG 文档根目录。
    root = Path(root_dir) if root_dir else DEFAULT_RAG_DOCS_DIR

    # 2. 在读取前检查目录是否存在，避免后面出现难理解的空索引错误。
    if not root.exists():
        raise FileNotFoundError(f"RAG 文档目录不存在: {root}")

    # 3. 递归寻找所有 Markdown 文件，并排序保证每次加载顺序稳定。
    markdown_files = sorted(root.rglob("*.md"))

    if not markdown_files:
        raise ValueError(f"RAG 文档目录中没有 Markdown 文件: {root}")

    # 4. 创建统一文本切分器。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )

    documents: list[Document] = []
    doc_id_to_source: dict[str, str] = {}

    # 5. 逐文件解析 frontmatter、section 和 chunks。
    for path in markdown_files:
        file_documents = _load_one_markdown_file(
            path=path,
            root_dir=root,
            splitter=splitter,
        )

        # 6. 检查 doc_id 是否被两个不同文件重复使用。
        #    doc_id 是稳定业务标识，同一个 doc_id 指向多个文件会导致 chunk_id 冲突。
        if file_documents:
            doc_id = str(file_documents[0].metadata["doc_id"])
            source = str(file_documents[0].metadata["source"])
            previous_source = doc_id_to_source.get(doc_id)

            if previous_source is not None and previous_source != source:
                raise ValueError(
                    f"RAG doc_id 重复: {doc_id}; "
                    f"sources={previous_source}, {source}"
                )

            doc_id_to_source[doc_id] = source

        documents.extend(file_documents)

    return documents


def deduplicate_documents(
    documents: Iterable[Document],
) -> DeduplicationResult:
    """
    根据 document_hash 做“文档级”去重。

    为什么不是只根据 chunk_hash 去重？
        如果两个 Markdown 文件内容完全相同，它们切出来的所有 chunks 也会相同。
        文档级去重可以一次性跳过整个重复文件，并保留清晰的重复来源报告。

    去重后返回的 documents 会同时用于：
        - 应用启动时构建一次 BM25。
        - 同步到持久化 Chroma。

    因此 BM25 和 Vector 使用的是同一批 chunks，不会出现两个检索器语料不一致。
    """

    # 1. 按 source 把同一个 Markdown 文件产生的 chunks 分组。
    chunks_by_source: dict[str, list[Document]] = {}

    for document in documents:
        source = str(document.metadata.get("source") or "")

        if not source:
            raise ValueError("RAG Document 缺少 metadata.source")

        chunks_by_source.setdefault(source, []).append(document)

    kept_documents: list[Document] = []
    duplicates: list[DuplicateDocument] = []
    hash_to_kept_source: dict[str, str] = {}

    # 2. 按 source 排序，保证重复文档总是稳定保留同一个文件。
    for source in sorted(chunks_by_source):
        source_chunks = chunks_by_source[source]

        if not source_chunks:
            continue

        document_hash = str(source_chunks[0].metadata.get("document_hash") or "")

        if not document_hash:
            raise ValueError(f"RAG Document 缺少 document_hash: {source}")

        # 3. 同一个 source 的所有 chunks 必须拥有相同 document_hash。
        if any(
            str(chunk.metadata.get("document_hash") or "") != document_hash
            for chunk in source_chunks
        ):
            raise ValueError(f"同一个 source 的 chunks 存在不同 document_hash: {source}")

        kept_source = hash_to_kept_source.get(document_hash)

        # 4. 如果 document_hash 已出现，说明整个文件重复，记录后跳过。
        if kept_source is not None:
            duplicates.append(
                DuplicateDocument(
                    kept_source=kept_source,
                    duplicate_source=source,
                    document_hash=document_hash,
                )
            )
            continue

        # 5. 第一次出现的 document_hash 被保留。
        hash_to_kept_source[document_hash] = source
        kept_documents.extend(source_chunks)

    return DeduplicationResult(
        documents=kept_documents,
        duplicates=duplicates,
    )


def _load_one_markdown_file(
    path: Path,
    root_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    加载、解析并切分一个 Markdown 文件。
    """

    # 1. 读取 UTF-8 文本，并移除可能存在的 BOM。
    raw_text = path.read_text(encoding="utf-8").lstrip("\ufeff")

    # 2. 将 YAML frontmatter 和正文分开。
    frontmatter, body = _parse_frontmatter(raw_text)

    # 3. 当 frontmatter 缺少基础 metadata 时，从目录和文件名做保守推断。
    inferred_metadata = _infer_metadata(path=path, body=body)

    # 4. 显式 frontmatter 优先级高于推断 metadata。
    file_metadata = {
        **inferred_metadata,
        **{
            key: value
            for key, value in frontmatter.items()
            if value is not None
        },
    }

    doc_type = file_metadata.get("doc_type")

    if not doc_type:
        raise ValueError(f"无法确定 RAG 文档 doc_type: {path}")

    source = path.relative_to(root_dir).as_posix()
    doc_id = str(file_metadata.get("doc_id") or f"{doc_type}_{path.stem}")
    title = str(file_metadata.get("title") or path.stem)

    # 5. 对“规范化正文 + 语义 metadata”计算 SHA-256 文档指纹。
    #    SHA-256 比 MD5 拥有更强的抗碰撞能力，并且 Python 标准库原生支持。
    document_hash = _build_document_hash(
        body=body,
        metadata=file_metadata,
    )

    # 6. 先按 Markdown 标题划分 section，使每个 chunk 都能保留 section 来源。
    sections = _split_markdown_sections(
        body=body,
        fallback_title=title,
    )

    output: list[Document] = []

    for section_index, (section_title, section_text) in enumerate(
        sections,
        start=1,
    ):
        # 7. 在 section 内执行字符切块。
        chunks = splitter.split_text(section_text)

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            normalized_chunk = chunk_text.strip()

            if not normalized_chunk:
                continue

            # 8. chunk_id 使用稳定 doc_id + section 序号 + chunk 序号。
            #    同一个文件内容更新时，已有位置可以 update；减少 chunk 时，旧 ID 会被同步逻辑删除。
            chunk_id = f"{doc_id}::{section_index:02d}::{chunk_index:03d}"

            # 9. chunk_hash 用来判断向量库中的单个 chunk 是否真的发生变化。
            chunk_hash = _build_chunk_hash(
                chunk_text=normalized_chunk,
                document_hash=document_hash,
                section=section_title,
            )

            metadata = {
                **file_metadata,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": source,
                "title": title,
                "section": section_title,
                "document_hash": document_hash,
                "chunk_hash": chunk_hash,
                # 10. 保留旧代码使用的 content_hash 字段，避免已写测试或后续代码失效。
                "content_hash": chunk_hash,
            }

            output.append(
                Document(
                    page_content=normalized_chunk,
                    metadata=metadata,
                )
            )

    return output


def _build_document_hash(
    body: str,
    metadata: Mapping[str, Any],
) -> str:
    """
    计算文档级 SHA-256 指纹。

    指纹内容包含：
        - 规范化正文。
        - 影响检索语义的 metadata。

    不包含：
        - source 文件路径。
        - doc_id。
        - title。

    因此同一内容即使复制到另一个文件名，也会被识别为重复文档。
    """

    semantic_metadata = {
        key: _normalize_hash_value(metadata.get(key))
        for key in SEMANTIC_METADATA_KEYS
        if metadata.get(key) is not None
    }

    # 1. 去掉文档开头的一级标题。
    #    一级标题通常只是 title 的 Markdown 展示形式；如果两个文件只有 doc_id、
    #    文件名和一级标题不同，但正文与语义 metadata 完全相同，应识别为重复文档。
    semantic_body = _remove_leading_document_title(body)

    # 2. 使用规范化正文和语义 metadata 生成稳定 JSON。
    payload = {
        "body": _normalize_text_for_hash(semantic_body),
        "metadata": semantic_metadata,
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def _remove_leading_document_title(body: str) -> str:
    """
    移除正文开头的第一个 Markdown 一级标题。

    为什么需要这一步？
        frontmatter.title、doc_id 和文件名属于展示/管理信息，不应该让一份完全
        相同的正文因为标题名称略有不同就绕过去重。

    只移除：
        文档开头第一个非空行，且该行形如 ``# 标题``。

    不移除：
        后续 ``## section`` 标题，因为 section 标题属于正文语义的一部分。
    """

    # 1. 保留原始换行结构，找到第一个非空行。
    lines = body.splitlines()
    first_content_index: int | None = None

    for index, line in enumerate(lines):
        if line.strip():
            first_content_index = index
            break

    # 2. 空正文直接返回。
    if first_content_index is None:
        return body

    first_line = lines[first_content_index].strip()

    # 3. 只识别一级标题；``##`` 及更深层标题不能删除。
    if re.match(r"^#(?!#)\s+.+$", first_line):
        del lines[first_content_index]

    # 4. 重新拼接正文，后续统一空白会由 _normalize_text_for_hash() 处理。
    return "\n".join(lines)

def _build_chunk_hash(
    chunk_text: str,
    document_hash: str,
    section: str,
) -> str:
    """
    计算 chunk 级 SHA-256 指纹。

    document_hash 和 section 也参与计算，避免相同句子在不同语义文档中被错误视为同一个 chunk。
    """

    payload = {
        "document_hash": document_hash,
        "section": section.strip(),
        "content": _normalize_text_for_hash(chunk_text),
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_text_for_hash(text: str) -> str:
    """
    规范化文本后再计算指纹。

    目前只统一空白字符，不改变中文标点和大小写，避免过度归一化造成误判。
    """

    return re.sub(r"\s+", " ", text).strip()


def _normalize_hash_value(value: Any) -> Any:
    """
    将 metadata 转成稳定、可 JSON 序列化的形式。
    """

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_hash_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, (list, tuple, set)):
        normalized_items = [_normalize_hash_value(item) for item in value]
        return sorted(normalized_items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))

    return value


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    解析 Markdown 顶部的 YAML frontmatter。
    """

    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index: int | None = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        raise ValueError("Markdown frontmatter 缺少结束标记 ---")

    yaml_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])
    metadata = yaml.safe_load(yaml_text) or {}

    if not isinstance(metadata, dict):
        raise ValueError("Markdown frontmatter 必须解析为 dict")

    return metadata, body


def _infer_metadata(path: Path, body: str) -> dict[str, Any]:
    """
    当 frontmatter 不完整时，从目录名、文件名和少量正文推断基础 metadata。
    """

    parent_name = path.parent.name
    doc_type = DIRECTORY_DOC_TYPE_MAP.get(parent_name)
    searchable_text = f"{path.name} {body[:300]}".lower()
    city = None

    for alias, normalized_city in CITY_ALIASES.items():
        if alias.lower() in searchable_text:
            city = normalized_city
            break

    return {
        "doc_type": doc_type,
        "city": city,
    }


def _split_markdown_sections(
    body: str,
    fallback_title: str,
) -> list[tuple[str, str]]:
    """
    按 Markdown 标题切分 section，再交给字符切分器。
    """

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    sections: list[tuple[str, str]] = []
    current_title = fallback_title
    current_lines: list[str] = []

    for line in body.splitlines():
        match = heading_pattern.match(line)

        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))

            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, text) for title, text in sections if text]

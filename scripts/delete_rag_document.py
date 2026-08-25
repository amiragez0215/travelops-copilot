from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1. 允许直接运行脚本时导入项目根目录下的 app 包。
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

# 2. 所有相对路径都以项目根目录为基准，并读取当前 .env。
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env", override=False)

from app.rag.runtime import initialize_hybrid_retriever  # noqa: E402


def main() -> None:
    """
    删除一个 RAG source 对应的 Chroma 记录和当前进程内 BM25 chunks。

    只删除索引记录：
        python scripts/delete_rag_document.py --source hotel_reviews/hotel_001.md

    同时删除 Markdown 源文件：
        python scripts/delete_rag_document.py `
            --source hotel_reviews/hotel_001.md `
            --delete-source-file

    注意：
        只删除 Chroma 而保留 Markdown 时，下次同步会把文档重新写入向量库。
        需要永久删除时必须同时删除 Markdown 源文件。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="相对于 data/rag_docs 的 source 路径。",
    )
    parser.add_argument(
        "--delete-source-file",
        action="store_true",
        help="同时删除 Markdown 文件，使删除在应用重启后仍然有效。",
    )
    args = parser.parse_args()

    # 3. 加载当前 Chroma、文档集合和应用内 BM25。
    retriever = initialize_hybrid_retriever()

    # 4. 统一删除 Chroma 与内存 BM25 中的同一 source，避免两路检索不一致。
    result = retriever.delete_source(
        source=args.source,
        delete_source_file=args.delete_source_file,
    )

    # 5. 输出删除数量和源文件是否已删除。
    pprint(result)


if __name__ == "__main__":
    main()

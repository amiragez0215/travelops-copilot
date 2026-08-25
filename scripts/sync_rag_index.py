from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv


# 项目根目录：travelops-copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1. 允许通过 `python scripts/sync_rag_index.py` 直接运行脚本。
#    Python 执行脚本时默认把 scripts/ 放进 sys.path，未必能找到 app/。
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

# 2. 相对路径统一以项目根目录为基准，并显式加载 .env。
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env", override=False)

from app.rag.runtime import initialize_hybrid_retriever  # noqa: E402


def main() -> None:
    """
    加载 RAG Markdown、SHA-256 去重、同步 Chroma，并构建一次 BM25。

    正常使用：
        python scripts/sync_rag_index.py

    更换 Embedding 模型或向量维度后完整重建：
        python scripts/sync_rag_index.py --reset-collection
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help=(
            "删除现有 Chroma collection 后重新写入全部 chunks。"
            "只有更换 Embedding 模型、向量维度或大幅调整切块规则时才使用。"
        ),
    )
    args = parser.parse_args()

    # 3. force_reload=True 表示重新读取 Markdown，并重建当前进程内的 BM25。
    #    ChromaVectorStore 会根据 chunk_hash 只编码新增或变化的 chunks。
    retriever = initialize_hybrid_retriever(
        force_reload=True,
        reset_collection=args.reset_collection,
    )

    # 4. 打印新增、更新、跳过和删除数量，便于确认增量同步结果。
    pprint(retriever.startup_report.to_dict())


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# 当前文件：
# D:/agent/travelops-copilot/scripts/check_runtime_assets.py
#
# parents[0] = scripts
# parents[1] = travelops-copilot
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 当使用：
# python scripts/check_runtime_assets.py
#
# 运行脚本时，Python 默认只会把 scripts 目录加入 sys.path，
# 不一定能找到项目根目录下的 app 包。
#
# 所以这里显式把项目根目录放到模块搜索路径最前面。
project_root_string = str(PROJECT_ROOT)

if project_root_string not in sys.path:
    sys.path.insert(0, project_root_string)


def _resolve_project_path(value: str) -> Path:
    """
    将 .env 中的模型路径解析成绝对路径。

    支持两种写法：

        相对路径：
            ./models/bge-reranker-v2-m3

        绝对路径：
            D:/agent/travelops-copilot/models/bge-reranker-v2-m3
    """

    path = Path(value)

    # 1. 相对路径以项目根目录为基准。
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # 2. resolve() 将路径标准化为绝对路径。
    return path.resolve()


def _read_required_env(name: str) -> str:
    """
    读取必须存在且不能为空的环境变量。
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"环境变量 {name} 为空，请检查项目根目录 .env。"
        )

    return value.strip()


def _check_model_directory(
    env_name: str,
) -> Path:
    """
    检查本地模型目录是否存在。

    这里不假设权重一定叫 model.safetensors，
    因为模型也可能使用分片权重。
    """

    raw_path = _read_required_env(
        env_name
    )

    model_path = _resolve_project_path(
        raw_path
    )

    # 1. 检查路径存在。
    if not model_path.exists():
        raise FileNotFoundError(
            f"{env_name} 指向的路径不存在：{model_path}"
        )

    # 2. 检查它是目录而不是普通文件。
    if not model_path.is_dir():
        raise NotADirectoryError(
            f"{env_name} 必须指向模型目录：{model_path}"
        )

    # 3. Hugging Face 模型一般至少需要 config.json。
    config_file = model_path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"模型目录缺少 config.json：{config_file}"
        )

    return model_path


def _check_embedding_model() -> None:
    """
    加载真实本地 Embedding 模型，并编码一个中文查询。

    这一步验证：

        1. RAG_EMBEDDING_MODEL 路径正确。
        2. sentence-transformers 模型文件完整。
        3. 当前 PyTorch 环境能够加载模型。
        4. embed_query() 可以返回向量。
    """

    # 1. 在加载 .env 后才导入 Embedding 工厂。
    #    避免模块导入过早导致环境变量没有生效。
    from app.rag.embeddings import (
        create_default_embeddings,
    )

    embeddings = create_default_embeddings()

    # 2. 使用真实模型编码一条中文查询。
    vector = embeddings.embed_query(
        "成都雨天三日游，美食和慢节奏安排"
    )

    # 3. 检查返回值是否是非空向量。
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(
            "Embedding 模型没有返回有效向量。"
        )

    print(
        "[OK] Embedding model loaded: "
        f"dimension={len(vector)}"
    )


def _check_rerank_model() -> None:
    """
    加载真实本地 Reranker，并对一个 Query-Passage 文本对评分。

    这一步验证：

        1. RAG_RERANK_MODEL 路径正确。
        2. tokenizer 和模型权重完整。
        3. Cross-Encoder 可以在当前设备运行。
        4. score_pairs() 可以返回相关性分数。
    """

    from app.rag.rerank_runtime import (
        initialize_reranker,
    )

    # 1. force_reload=True，保证本次确实读取当前 .env 配置。
    reranker = initialize_reranker(
        force_reload=True
    )

    # 2. 对一条 Query-Passage 执行真实模型推理。
    scores = reranker.scorer.score_pairs(
        [
            (
                "成都雨天三日游需要准备什么？",
                "雨天旅行建议携带雨具、防滑鞋和备用袜子。",
            )
        ]
    )

    # 3. 检查分数数量和内容。
    if len(scores) != 1:
        raise RuntimeError(
            "Reranker 返回的分数数量不正确。"
        )

    score = scores[0]

    print(
        "[OK] Rerank model loaded: "
        f"raw_score={score.raw_score:.6f}, "
        f"normalized_score={score.normalized_score:.6f}"
    )


def main() -> None:
    """
    检查项目运行时资源。

    这个脚本不属于普通单元测试。
    只在以下场景运行：

        - 第一次下载本地模型后。
        - 修改模型目录后。
        - 更换 Python 环境后。
        - 模型加载报错时。
    """

    # 1. 将当前工作目录切换到项目根目录。
    #    这样 ./models、./data 等相对路径都能稳定解析。
    os.chdir(PROJECT_ROOT)

    # 2. 显式加载项目根目录 .env。
    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )

    # 3. 检查 .env 中的两个本地模型目录。
    embedding_path = _check_model_directory(
        "RAG_EMBEDDING_MODEL"
    )

    rerank_path = _check_model_directory(
        "RAG_RERANK_MODEL"
    )

    print(
        "[OK] Embedding path: "
        f"{embedding_path}"
    )

    print(
        "[OK] Rerank path: "
        f"{rerank_path}"
    )

    # 4. 只检查 DeepSeek Key 是否配置。
    #    不输出 Key 内容，避免秘密泄漏。
    deepseek_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    print(
        "[INFO] DeepSeek key configured: "
        f"{bool(deepseek_key)}"
    )

    # 5. 加载并运行真实 Embedding 模型。
    _check_embedding_model()

    # 6. 加载并运行真实 Reranker。
    _check_rerank_model()

    print(
        "\nAll local runtime assets are ready."
    )


if __name__ == "__main__":
    main()
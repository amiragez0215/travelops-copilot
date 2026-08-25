from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_DIR = PROJECT_ROOT / "data" / "mock"


def resolve_mock_file(
    filename: str,
    mock_file: str | Path | None = None,
) -> Path:
    """
    解析 mock 文件路径。

    Args:
        filename:
            默认 mock 文件名，例如 weather_mock.json。

        mock_file:
            测试时可以传入临时 mock 文件路径。
            如果为 None，则使用 data/mock/<filename>。

    Returns:
        Path:
            最终 mock 文件路径。
    """

    if mock_file is not None:
        return Path(mock_file)

    return DEFAULT_MOCK_DIR / filename


def load_mock_json(
    filename: str,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    读取 mock JSON 文件。

    当前 mock 数据统一约定：
        {
            "provider": "...",
            "version": 1,
            "items": [...]
        }

    这个函数只负责读取和基础 JSON 校验，不做业务过滤。
    """

    path = resolve_mock_file(filename=filename, mock_file=mock_file)

    if not path.exists():
        raise FileNotFoundError(f"mock 文件不存在: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"mock 文件根节点必须是 dict: {path}")

    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"mock 文件必须包含 items 列表: {path}")

    return data
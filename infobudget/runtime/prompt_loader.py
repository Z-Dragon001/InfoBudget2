"""功能：加载外置 Prompt 模板。
输入：Prompt 目录与模板名称。
输出：模板字符串。
依赖：pathlib。
作者：OpenAI Codex
"""

from __future__ import annotations

from pathlib import Path


def load_prompt(prompt_dir: str | Path, name: str) -> str:
    """读取 Prompt 模板文件。"""
    path = Path(prompt_dir) / name
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()

"""功能：提供统一日志初始化。
输入：日志名称。
输出：配置好的 logger。
依赖：logging。
作者：OpenAI Codex
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger。"""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logger

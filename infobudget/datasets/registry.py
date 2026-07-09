"""功能：注册支持的数据集预处理器。
输入：数据集名称。
输出：对应预处理器实例。
依赖：datasets.locomo、datasets.longmemeval。
作者：OpenAI Codex
"""

from __future__ import annotations

from infobudget.datasets.base import BaseDatasetPreprocessor
from infobudget.datasets.longmemeval import LongMemEvalPreprocessor
from infobudget.datasets.locomo import LOCOMOPreprocessor


class DatasetRegistry:
    """数据集预处理器注册表。"""

    _REGISTRY: dict[str, type[BaseDatasetPreprocessor]] = {
        "locomo": LOCOMOPreprocessor,
        "longmemeval": LongMemEvalPreprocessor,
    }

    @classmethod
    def create(cls, dataset_name: str) -> BaseDatasetPreprocessor:
        normalized = dataset_name.lower()
        if normalized not in cls._REGISTRY:
            raise KeyError(f"Unsupported dataset: {dataset_name}")
        return cls._REGISTRY[normalized]()

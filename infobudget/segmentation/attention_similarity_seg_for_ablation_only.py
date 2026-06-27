"""功能：保留 attention segmentation 的占位接口。
输入：Turn 列表。
输出：未实现异常。
依赖：schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from infobudget.schemas import Segment, Turn
from infobudget.segmentation.base import BaseSegmenter


class AttentionSimilaritySegForAblationOnly(BaseSegmenter):
    """仅用于未来消融实验的占位类。"""

    def segment(self, turns: list[Turn]) -> list[Segment]:
        raise NotImplementedError(
            "attention_similarity_seg_for_ablation_only is deferred in InfoBudget v1.0"
        )

    def segment_with_trace(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        raise NotImplementedError(
            "attention_similarity_seg_for_ablation_only is deferred in InfoBudget v1.0"
        )

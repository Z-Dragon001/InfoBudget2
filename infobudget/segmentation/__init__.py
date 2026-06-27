"""功能：导出主题分段组件。
输入：分段器导入请求。
输出：可复用分段器类。
依赖：项目分段模块。
作者：OpenAI Codex
"""

from infobudget.segmentation.attention_similarity_seg_for_ablation_only import (
    AttentionSimilaritySegForAblationOnly,
)
from infobudget.segmentation.fixed_window import FixedWindowSeg
from infobudget.segmentation.lite_topic_seg import LiteTopicSeg
from infobudget.segmentation.similarity_only import SimilarityOnlySeg

__all__ = [
    "AttentionSimilaritySegForAblationOnly",
    "FixedWindowSeg",
    "LiteTopicSeg",
    "SimilarityOnlySeg",
]

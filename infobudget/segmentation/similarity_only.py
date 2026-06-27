"""功能：提供仅基于相似度阈值的分段 baseline。
输入：Turn 列表。
输出：Segment 列表与 trace。
依赖：配置、LiteTopicSeg。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import asdict

from infobudget.config import SegmentationConfig
from infobudget.segmentation.lite_topic_seg import LiteTopicSeg


class SimilarityOnlySeg(LiteTopicSeg):
    """禁用 drop 检测的简化分段器。"""

    def __init__(self, cfg: SegmentationConfig):
        cfg = SegmentationConfig(**{**asdict(cfg), "drop_threshold": 1.1})
        super().__init__(cfg)

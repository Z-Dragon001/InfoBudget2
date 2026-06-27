"""功能：提供固定窗口分段 baseline。
输入：Turn 列表。
输出：按固定轮数切分的 Segment 列表。
依赖：schemas、text。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.schemas import Segment, Turn
from infobudget.segmentation.base import BaseSegmenter


@dataclass(slots=True)
class FixedWindowSeg(BaseSegmenter):
    """固定窗口分段 baseline。"""

    window_size: int = 6

    def segment(self, turns: list[Turn]) -> list[Segment]:
        segments, _ = self.segment_with_trace(turns)
        return segments

    def segment_with_trace(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        segments: list[Segment] = []
        for index in range(0, len(turns), self.window_size):
            chunk = turns[index : index + self.window_size]
            segments.append(
                Segment(
                    segment_id=f"seg_{len(segments)+1:06d}",
                    start_turn=chunk[0].turn_id,
                    end_turn=chunk[-1].turn_id,
                    turn_ids=[item.turn_id for item in chunk],
                    text="\n".join(f"{item.role}: {item.text}" for item in chunk),
                    token_count=sum(item.token_count for item in chunk),
                    mean_adjacent_similarity=0.0,
                    boundary_reason="fixed_window",
                )
            )
        return segments, {"method": "fixed_window_seg", "window_size": self.window_size}

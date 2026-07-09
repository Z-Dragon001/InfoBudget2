"""功能：实现 LiteTopicSeg 轻量主题分割。
输入：Turn 列表与分段配置。
输出：Segment 列表及 trace。
依赖：numpy、配置、编码器。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from infobudget.config import SegmentationConfig
from infobudget.schemas import Segment, Turn
from infobudget.segmentation.base import BaseSegmenter
from infobudget.utils.embeddings import HashingTextEncoder, TextEncoder, cosine_similarity
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _SegmentState:
    start: int
    end: int
    reason: str


class LiteTopicSeg(BaseSegmenter):
    """默认的 embedding-only 分段器。"""

    def __init__(self, cfg: SegmentationConfig, encoder: TextEncoder | None = None):
        self.cfg = cfg
        self.encoder = encoder or HashingTextEncoder()

    def segment(self, turns: list[Turn]) -> list[Segment]:
        segments, _ = self.segment_with_trace(turns)
        return segments

    def segment_with_trace(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        if not turns:
            return [], {"method": "lite_topic_seg", "reason": "empty_input"}
        if len(turns) == 1:
            segment = self._make_segment(turns, 0, 0, "single_turn", [1.0])
            return self.reindex_segments([segment]), {"method": "lite_topic_seg", "boundaries": [1]}

        embeddings = self.encoder.encode_batch([turn.memory_text() for turn in turns])
        similarities = [cosine_similarity(embeddings[i - 1], embeddings[i]) for i in range(1, len(turns))]
        smoothed = self._smooth(similarities)
        boundaries, reasons = self._detect_boundaries(smoothed)
        states = self._build_states(turns, boundaries, reasons, embeddings, smoothed)
        states = self._merge_short(states, turns, embeddings)
        states = self._split_overlong(states, turns, smoothed)
        segments = [
            self._make_segment(turns, state.start, state.end, state.reason, similarities)
            for state in states
        ]
        trace = {
            "method": "lite_topic_seg",
            "similarities": similarities,
            "smoothed_similarities": smoothed,
            "boundaries": boundaries,
            "num_segments": len(segments),
        }
        logger.info("LiteTopicSeg produced %s segments", len(segments))
        return self.reindex_segments(segments), trace

    def _smooth(self, similarities: list[float]) -> list[float]:
        if not similarities:
            return []
        if not self.cfg.enable_smoothing:
            return similarities[:]
        smoothed = [similarities[0]]
        for value in similarities[1:]:
            smoothed.append(self.cfg.smooth_alpha * value + (1 - self.cfg.smooth_alpha) * smoothed[-1])
        return smoothed

    def _detect_boundaries(self, similarities: list[float]) -> tuple[list[int], dict[int, str]]:
        boundaries = [1]
        reasons: dict[int, str] = {1: "start"}
        candidates: list[tuple[int, float, str]] = []
        for idx, similarity in enumerate(similarities, start=2):
            low_sim = similarity < self.cfg.similarity_threshold
            drop = idx > 2 and (similarities[idx - 3] - similarity > self.cfg.drop_threshold)
            if low_sim or drop:
                reason = "low_similarity+drop" if low_sim and drop else "low_similarity" if low_sim else "similarity_drop"
                candidates.append((idx, similarity, reason))
        pruned = self._prune_dense(candidates)
        for boundary, _score, reason in pruned:
            boundaries.append(boundary)
            reasons[boundary] = reason
        return sorted(boundaries), reasons

    def _prune_dense(self, candidates: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
        if not candidates:
            return []
        kept: list[tuple[int, float, str]] = [candidates[0]]
        for current in candidates[1:]:
            previous = kept[-1]
            if current[0] - previous[0] < self.cfg.min_boundary_gap:
                if current[1] < previous[1]:
                    kept[-1] = current
            else:
                kept.append(current)
        return kept

    def _build_states(
        self,
        turns: list[Turn],
        boundaries: list[int],
        reasons: dict[int, str],
        embeddings: np.ndarray,
        smoothed: list[float],
    ) -> list[_SegmentState]:
        states: list[_SegmentState] = []
        turn_count = len(turns)
        starts = boundaries + [turn_count + 1]
        for left, right in zip(starts, starts[1:]):
            states.append(_SegmentState(start=left - 1, end=right - 2, reason=reasons[left]))
        return states

    def _merge_short(
        self,
        states: list[_SegmentState],
        turns: list[Turn],
        embeddings: np.ndarray,
    ) -> list[_SegmentState]:
        if not self.cfg.merge_short_segment or len(states) <= 1:
            return states
        merged = states[:]
        changed = True
        while changed and len(merged) > 1:
            changed = False
            for index, state in enumerate(list(merged)):
                turn_size = state.end - state.start + 1
                token_size = sum(turns[i].token_count for i in range(state.start, state.end + 1))
                if turn_size >= self.cfg.min_segment_turns and token_size >= self.cfg.min_segment_tokens:
                    continue
                left_score = -1.0
                right_score = -1.0
                if index > 0:
                    left_score = self._centroid_similarity(merged[index - 1], state, embeddings)
                if index < len(merged) - 1:
                    right_score = self._centroid_similarity(state, merged[index + 1], embeddings)
                target_index = index - 1 if left_score >= right_score else index + 1
                if target_index < 0 or target_index >= len(merged):
                    continue
                left = min(state.start, merged[target_index].start)
                right = max(state.end, merged[target_index].end)
                if right - left + 1 > self.cfg.max_segment_turns:
                    continue
                merged[min(index, target_index)] = _SegmentState(left, right, "merged_short")
                del merged[max(index, target_index)]
                changed = True
                break
        return merged

    def _centroid_similarity(
        self,
        left: _SegmentState,
        right: _SegmentState,
        embeddings: np.ndarray,
    ) -> float:
        left_vec = embeddings[left.start : left.end + 1].mean(axis=0)
        right_vec = embeddings[right.start : right.end + 1].mean(axis=0)
        return cosine_similarity(left_vec, right_vec)

    def _split_overlong(
        self,
        states: list[_SegmentState],
        turns: list[Turn],
        smoothed: list[float],
    ) -> list[_SegmentState]:
        result: list[_SegmentState] = []
        for state in states:
            stack = [state]
            while stack:
                current = stack.pop()
                turn_size = current.end - current.start + 1
                token_size = sum(turns[i].token_count for i in range(current.start, current.end + 1))
                if turn_size <= self.cfg.max_segment_turns and token_size <= self.cfg.max_segment_tokens:
                    result.append(current)
                    continue
                if turn_size <= 1:
                    result.append(current)
                    continue
                split_at = self._choose_split_point(current, turns, smoothed)
                stack.append(_SegmentState(split_at, current.end, "forced_split"))
                stack.append(_SegmentState(current.start, split_at - 1, "forced_split"))
        return sorted(result, key=lambda item: item.start)

    def _choose_split_point(self, state: _SegmentState, turns: list[Turn], smoothed: list[float]) -> int:
        candidates = list(range(state.start + 1, state.end + 1))
        ranked = []
        total_tokens = sum(turns[i].token_count for i in range(state.start, state.end + 1))
        for split in candidates:
            left_tokens = sum(turns[i].token_count for i in range(state.start, split))
            right_tokens = total_tokens - left_tokens
            similarity = smoothed[split - 1] if split - 1 < len(smoothed) else 1.0
            ranked.append((similarity, abs(left_tokens - right_tokens), split))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return ranked[0][2]

    def _make_segment(
        self,
        turns: list[Turn],
        start: int,
        end: int,
        reason: str,
        similarities: list[float],
    ) -> Segment:
        chunk = turns[start : end + 1]
        local_sims: list[float] = []
        for idx in range(start + 1, end + 1):
            if idx - 1 < len(similarities):
                local_sims.append(similarities[idx - 1])
        mean_similarity = float(np.mean(local_sims)) if local_sims else 1.0
        return Segment(
            segment_id="",
            start_turn=chunk[0].turn_id,
            end_turn=chunk[-1].turn_id,
            turn_ids=[item.turn_id for item in chunk],
            text="\n".join(item.memory_line() for item in chunk),
            token_count=sum(item.token_count for item in chunk),
            mean_adjacent_similarity=mean_similarity,
            boundary_reason=reason,
        )

    @staticmethod
    def reindex_segments(segments: list[Segment]) -> list[Segment]:
        """重建 segment id。"""
        for index, segment in enumerate(segments, start=1):
            segment.segment_id = f"seg_{index:06d}"
        return segments

    def __call__(self, turns: list[Turn]) -> list[Segment]:
        """便捷调用。"""
        return self.reindex_segments(self.segment(turns))

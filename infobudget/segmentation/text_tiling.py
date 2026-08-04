"""Shared TextTiling boundary selection and Segment construction."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from infobudget.config import SegmentationConfig
from infobudget.schemas import Segment, Turn
from infobudget.segmentation.base import BaseSegmenter
from infobudget.segmentation.session_utils import contiguous_session_slices

PairScorer = Callable[[list[str]], list[float]]


@dataclass(slots=True)
class _SegmentState:
    start: int
    end: int
    reason: str


class TextTilingSegmenter(BaseSegmenter):
    """Base class for BERT pair scorers followed by TextTiling."""

    method_name = "text_tiling"
    boundary_reason = "texttiling_depth"

    def __init__(self, cfg: SegmentationConfig, pair_scorer: PairScorer | None = None):
        self.cfg = cfg
        self._pair_scorer = pair_scorer

    def segment(self, turns: list[Turn]) -> list[Segment]:
        segments, _ = self.segment_with_trace(turns)
        return segments

    def segment_with_trace(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        if not turns:
            return [], {"method": self.method_name, "reason": "empty_input"}

        segments: list[Segment] = []
        session_traces: list[dict] = []
        for session_index, session_slice in enumerate(
            contiguous_session_slices(turns, self.cfg.preserve_session_boundaries)
        ):
            session_turns = turns[session_slice.start : session_slice.end]
            session_segments, trace = self._segment_session(session_turns)
            if session_index > 0 and session_segments:
                session_segments[0].boundary_reason = "session_boundary"
            segments.extend(session_segments)
            session_traces.append(
                {
                    "session_id": session_slice.session_id,
                    "start_index": session_slice.start,
                    "end_index": session_slice.end,
                    **trace,
                }
            )

        self.reindex_segments(segments)
        return segments, {
            "method": self.method_name,
            "alpha": self.cfg.adaptive_alpha,
            "sessions": session_traces,
            "num_segments": len(segments),
        }

    def _segment_session(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        if len(turns) == 1:
            return [self._make_segment(turns, 0, 0, "single_turn", [])], {
                "coherence_scores": [],
                "depth_scores": [],
                "threshold": None,
                "boundaries": [1],
            }

        utterances = [turn.memory_text() for turn in turns]
        coherence = self.score_adjacent_pairs(utterances)
        if len(coherence) != len(turns) - 1:
            raise ValueError(
                f"{self.method_name} returned {len(coherence)} scores for "
                f"{len(turns) - 1} adjacent pairs"
            )
        depths = texttiling_depth_scores(coherence)
        threshold = adaptive_threshold(depths, self.cfg.adaptive_alpha)
        candidate_splits = [index + 1 for index, value in enumerate(depths) if value > threshold]
        splits = self._prune_dense(candidate_splits, depths)
        states = self._states_from_splits(len(turns), splits)
        states = self._merge_short(states, turns, coherence)
        states = self._split_overlong(states, turns, coherence)
        segments = [
            self._make_segment(turns, state.start, state.end, state.reason, coherence)
            for state in states
        ]
        return segments, {
            "coherence_scores": coherence,
            "depth_scores": depths,
            "threshold": threshold,
            "boundaries": [1, *[split + 1 for split in splits]],
        }

    def score_adjacent_pairs(self, utterances: list[str]) -> list[float]:
        if self._pair_scorer is not None:
            return [float(value) for value in self._pair_scorer(utterances)]
        return self._score_adjacent_pairs(utterances)

    @abstractmethod
    def _score_adjacent_pairs(self, utterances: list[str]) -> list[float]:
        """Return one coherence probability per adjacent utterance pair."""

    def _prune_dense(self, splits: list[int], depths: list[float]) -> list[int]:
        if not splits:
            return []
        kept = [splits[0]]
        for split in splits[1:]:
            if split - kept[-1] < self.cfg.min_boundary_gap:
                if depths[split - 1] > depths[kept[-1] - 1]:
                    kept[-1] = split
            else:
                kept.append(split)
        return kept

    def _states_from_splits(self, turn_count: int, splits: list[int]) -> list[_SegmentState]:
        starts = [0, *splits]
        ends = [*splits, turn_count]
        return [
            _SegmentState(start, end - 1, "start" if index == 0 else self.boundary_reason)
            for index, (start, end) in enumerate(zip(starts, ends))
        ]

    def _merge_short(
        self,
        states: list[_SegmentState],
        turns: list[Turn],
        coherence: list[float],
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
                left_score = coherence[state.start - 1] if index > 0 else -1.0
                right_score = coherence[state.end] if index < len(merged) - 1 else -1.0
                target_index = index - 1 if left_score >= right_score else index + 1
                left = min(state.start, merged[target_index].start)
                right = max(state.end, merged[target_index].end)
                if right - left + 1 > self.cfg.max_segment_turns:
                    continue
                merged[min(index, target_index)] = _SegmentState(left, right, "merged_short")
                del merged[max(index, target_index)]
                changed = True
                break
        return merged

    def _split_overlong(
        self,
        states: list[_SegmentState],
        turns: list[Turn],
        coherence: list[float],
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
                split = self._choose_split_point(current, turns, coherence)
                stack.append(_SegmentState(split, current.end, "forced_split"))
                stack.append(_SegmentState(current.start, split - 1, "forced_split"))
        return sorted(result, key=lambda item: item.start)

    @staticmethod
    def _choose_split_point(
        state: _SegmentState,
        turns: list[Turn],
        coherence: list[float],
    ) -> int:
        total_tokens = sum(turns[i].token_count for i in range(state.start, state.end + 1))
        ranked: list[tuple[float, int, int]] = []
        for split in range(state.start + 1, state.end + 1):
            left_tokens = sum(turns[i].token_count for i in range(state.start, split))
            ranked.append((coherence[split - 1], abs(left_tokens - (total_tokens - left_tokens)), split))
        ranked.sort()
        return ranked[0][2]

    @staticmethod
    def _make_segment(
        turns: list[Turn],
        start: int,
        end: int,
        reason: str,
        coherence: list[float],
    ) -> Segment:
        chunk = turns[start : end + 1]
        local_scores = [coherence[index - 1] for index in range(start + 1, end + 1)]
        return Segment(
            segment_id="",
            start_turn=chunk[0].turn_id,
            end_turn=chunk[-1].turn_id,
            turn_ids=[turn.turn_id for turn in chunk],
            text="\n".join(turn.memory_line() for turn in chunk),
            token_count=sum(turn.token_count for turn in chunk),
            mean_adjacent_similarity=float(np.mean(local_scores)) if local_scores else 1.0,
            boundary_reason=reason,
        )

    @staticmethod
    def reindex_segments(segments: list[Segment]) -> list[Segment]:
        for index, segment in enumerate(segments, start=1):
            segment.segment_id = f"seg_{index:06d}"
        return segments


def texttiling_depth_scores(scores: list[float]) -> list[float]:
    """Compute local-valley depths from a coherence curve."""
    depths: list[float] = []
    for index, score in enumerate(scores):
        left_peak = score
        cursor = index - 1
        while cursor >= 0 and scores[cursor] >= left_peak:
            left_peak = scores[cursor]
            cursor -= 1

        right_peak = score
        cursor = index + 1
        while cursor < len(scores) and scores[cursor] >= right_peak:
            right_peak = scores[cursor]
            cursor += 1

        depths.append(float((left_peak + right_peak - 2.0 * score) / 2.0))
    return depths


def adaptive_threshold(depths: list[float], alpha: float) -> float:
    """Return tau = mean(depths) + alpha * std(depths)."""
    if not depths:
        return 0.0
    values = np.asarray(depths, dtype=np.float32)
    return float(values.mean() + alpha * values.std())

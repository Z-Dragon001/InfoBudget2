"""Tests for the two formal session-aware BERT segmentation modes."""

from __future__ import annotations

from dataclasses import replace

import pytest

from infobudget.config import load_project_bundle
from infobudget.schemas import Turn
from infobudget.segmentation.bert_mlp_text_tiling import BertMLPTextTilingSegmenter
from infobudget.segmentation.factory import build_segmenter
from infobudget.segmentation.nsp_text_tiling import NSPTextTilingSegmenter
from infobudget.segmentation.text_tiling import adaptive_threshold, texttiling_depth_scores


def _turn(turn_id: int, session_id: str) -> Turn:
    return Turn(
        turn_id,
        "user" if turn_id % 2 else "assistant",
        f"session text {turn_id}",
        20,
        f"2026-07-22T10:00:{turn_id:02d}.000",
        {"session_id": session_id, "weekday": "Wed"},
    )


def test_texttiling_depth_formula() -> None:
    scores = [0.90, 0.82, 0.20, 0.84, 0.88, 0.25, 0.86]
    depths = texttiling_depth_scores(scores)
    assert depths == pytest.approx([0.0, 0.04, 0.69, 0.02, 0.0, 0.62, 0.0])
    values = __import__("numpy").asarray(depths)
    assert adaptive_threshold(depths, 0.5) == pytest.approx(values.mean() + 0.5 * values.std())


@pytest.mark.parametrize("segmenter_type", [NSPTextTilingSegmenter, BertMLPTextTilingSegmenter])
def test_formal_segmenters_preserve_sessions(segmenter_type) -> None:
    cfg = replace(load_project_bundle("configs").config.segmentation, min_segment_tokens=1)
    turns = [*[_turn(i, "session_1") for i in range(1, 5)], *[_turn(i, "session_2") for i in range(5, 9)]]
    scorer = lambda _: [0.9, 0.1, 0.9]
    segmenter = (
        segmenter_type(cfg, "unused", pair_scorer=scorer)
        if segmenter_type is NSPTextTilingSegmenter
        else segmenter_type(cfg, "unused", "unused.pt", pair_scorer=scorer)
    )
    segments = segmenter.segment(turns)
    assert [item.turn_ids for item in segments] == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert segments[2].boundary_reason == "session_boundary"


def test_factory_exposes_only_two_formal_methods() -> None:
    bundle = load_project_bundle("configs")
    assert isinstance(build_segmenter(replace(bundle.config.segmentation, method="nsp_text_tiling"), bundle.root_dir), NSPTextTilingSegmenter)
    assert isinstance(build_segmenter(replace(bundle.config.segmentation, method="bert_mlp_text_tiling"), bundle.root_dir), BertMLPTextTilingSegmenter)
    with pytest.raises(ValueError, match="expected nsp_text_tiling or bert_mlp_text_tiling"):
        build_segmenter(replace(bundle.config.segmentation, method="lite_topic_seg"), bundle.root_dir)

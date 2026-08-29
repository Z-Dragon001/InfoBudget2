"""Tests for the two formal session-aware BERT segmentation modes."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from infobudget.config import load_project_bundle
from infobudget.schemas import DatasetDialogueExample, DatasetSession, Segment, Turn
from infobudget.segmentation.bert_mlp_text_tiling import BertMLPTextTilingSegmenter
from infobudget.segmentation.factory import build_segmenter
from infobudget.segmentation.identity import (
    adaptive_alpha_token,
    segmentation_artifact_name,
    segmentation_version,
)
from infobudget.segmentation.nsp_text_tiling import NSPTextTilingSegmenter
from infobudget.segmentation.pipeline import SegmentationRun
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


@pytest.mark.parametrize(
    ("alpha", "token"),
    [(0.5, "0p5"), (1.0, "1"), ("0.0500", "0p05"), (0, "0")],
)
def test_adaptive_alpha_has_a_stable_artifact_identity(alpha, token) -> None:
    assert adaptive_alpha_token(alpha) == token
    assert segmentation_artifact_name("nsp_text_tiling", alpha) == (
        f"nsp_text_tiling_alpha_{token}"
    )
    assert segmentation_version("nsp_text_tiling", alpha) == (
        f"nsp_text_tiling_alpha_{token}_v1"
    )


@pytest.mark.parametrize("alpha", [-0.1, float("inf"), float("nan")])
def test_adaptive_alpha_rejects_unsafe_values(alpha) -> None:
    with pytest.raises(ValueError, match="adaptive alpha"):
        adaptive_alpha_token(alpha)


def test_segmentation_run_writes_to_alpha_isolated_directory(
    tmp_path, monkeypatch
) -> None:
    bundle = load_project_bundle("configs")
    bundle.root_dir = tmp_path
    bundle.config.segmentation = replace(
        bundle.config.segmentation,
        method="nsp_text_tiling",
        adaptive_alpha=0.7,
    )
    processed = tmp_path / "datasets" / "processed" / "locomo" / "full"
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text("{}\n", encoding="utf-8")
    turns = [_turn(1, "session_1"), _turn(2, "session_1")]
    example = DatasetDialogueExample(
        sample_id="sample-1",
        dataset_name="locomo",
        split="full",
        sessions=[DatasetSession("session_1", None, None, turns)],
        dialogue=turns,
        qa_pairs=[],
    )

    class FakeLoader:
        def __init__(self, *_):
            pass

        def load(self, *_):
            return [example]

        def split_dir(self, *_):
            return processed

    class FakeSegmenter:
        def segment_with_trace(self, _):
            return (
                [Segment("pending", 1, 2, [1, 2], "one\ntwo", 40, 0.5, "end")],
                {},
            )

    monkeypatch.setattr("infobudget.segmentation.pipeline.DatasetLoader", FakeLoader)
    monkeypatch.setattr(
        "infobudget.segmentation.pipeline.build_segmenter", lambda *_: FakeSegmenter()
    )

    manifest = SegmentationRun(bundle).run("locomo", "full")
    output = (
        tmp_path
        / "datasets"
        / "segmented"
        / "locomo"
        / "full"
        / "nsp_text_tiling_alpha_0p7"
    )
    row = json.loads(
        (output / "samples" / "sample-1" / "segments.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert manifest["output_dir"] == str(output.resolve())
    assert manifest["segmentation_method"] == "nsp_text_tiling_alpha_0p7"
    assert manifest["segmentation_version"] == "nsp_text_tiling_alpha_0p7_v1"
    assert row["segmentation_method"] == "nsp_text_tiling_alpha_0p7"
    assert row["adaptive_alpha"] == pytest.approx(0.7)
    assert row["checkpoint_path"] is None

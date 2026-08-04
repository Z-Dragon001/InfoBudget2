from __future__ import annotations

import json
from pathlib import Path

import pytest

from infobudget.datasets.splits import load_split_selection


def _write_manifest(path, folds) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "conversation_split_v1",
                "dataset_name": "locomo",
                "source_split": "full",
                "protocol": "unit",
                "seed": 42,
                "source_processed_manifest_sha256": "unused-in-unit-test",
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_split_manifest_selects_only_the_requested_fold(tmp_path) -> None:
    path = tmp_path / "splits.json"
    _write_manifest(
        path,
        [
            {"fold": 0, "train": ["a", "b"], "validation": [], "test": ["c"]},
            {"fold": 1, "train": ["b", "c"], "validation": [], "test": ["a"]},
        ],
    )
    selected = load_split_selection(
        path,
        dataset_name="locomo",
        source_split="full",
        fold=1,
        available_sample_ids={"a", "b", "c"},
    )
    assert selected.train_sample_ids == ("b", "c")
    assert selected.test_sample_ids == ("a",)
    assert len(selected.sha256) == 64


def test_split_manifest_rejects_leakage_and_stale_coverage(tmp_path) -> None:
    path = tmp_path / "splits.json"
    _write_manifest(
        path,
        [{"fold": 0, "train": ["a", "b"], "validation": [], "test": ["b"]}],
    )
    with pytest.raises(ValueError, match="appears in both train and test"):
        load_split_selection(
            path,
            dataset_name="locomo",
            source_split="full",
            fold=0,
            available_sample_ids={"a", "b"},
        )

    _write_manifest(
        path,
        [{"fold": 0, "train": ["a"], "validation": [], "test": ["b"]}],
    )
    with pytest.raises(ValueError, match="omits available samples"):
        load_split_selection(
            path,
            dataset_name="locomo",
            source_split="full",
            fold=0,
            available_sample_ids={"a", "b", "c"},
        )


def test_bundled_locomo_cv5_holds_out_every_conversation_once() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "datasets" / "splits" / "locomo" / "cv5_seed42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = {
        path.name
        for path in (
            root / "datasets" / "segmented" / "locomo" / "full" / "nsp_text_tiling" / "samples"
        ).iterdir()
        if path.is_dir()
    }
    held_out = []
    for fold in range(5):
        selected = load_split_selection(
            path,
            dataset_name="locomo",
            source_split="full",
            fold=fold,
            available_sample_ids=available,
            source_processed_manifest_path=(
                root / "datasets" / "processed" / "locomo" / "full" / "manifest.json"
            ),
            project_root=root,
        )
        assert len(selected.train_sample_ids) == 8
        assert len(selected.test_sample_ids) == 2
        held_out.extend(selected.test_sample_ids)
    assert len(payload["folds"]) == 5
    assert len(held_out) == len(set(held_out)) == 10
    assert set(held_out) == available


def test_bundled_longmemeval_manifests_have_strict_sizes_and_evidence_isolation() -> None:
    root = Path(__file__).resolve().parents[1]
    split_root = root / "datasets" / "splits" / "longmemeval"
    available = {
        item.name
        for item in (
            root / "datasets" / "segmented" / "longmemeval" / "full" / "nsp_text_tiling" / "samples"
        ).iterdir()
        if item.is_dir()
    }
    source_manifest = root / "datasets" / "processed" / "longmemeval" / "full" / "manifest.json"

    fixed_path = split_root / "fixed_80_10_10_seed42_nsp_text_tiling.json"
    fixed = load_split_selection(
        fixed_path,
        dataset_name="longmemeval",
        source_split="full",
        fold=0,
        available_sample_ids=available,
        source_processed_manifest_path=source_manifest,
        project_root=root,
    )
    assert tuple(map(len, (fixed.train_sample_ids, fixed.validation_sample_ids, fixed.test_sample_ids))) == (
        400,
        50,
        50,
    )
    fixed_payload = json.loads(fixed_path.read_text(encoding="utf-8"))
    assert fixed_payload["folds"][0]["statistics"]["train"]["question_type"] == {
        "knowledge-update": 63,
        "multi-session": 106,
        "single-session-assistant": 45,
        "single-session-preference": 24,
        "single-session-user": 56,
        "temporal-reasoning": 106,
    }
    assert not any(
        fixed_payload["folds"][0]["audit"]["cross_partition_evidence_session_overlap"].values()
    )

    cv_path = split_root / "cv5_360_40_100_seed42_nsp_text_tiling.json"
    cv_payload = json.loads(cv_path.read_text(encoding="utf-8"))
    held_out = []
    for fold in range(5):
        selected = load_split_selection(
            cv_path,
            dataset_name="longmemeval",
            source_split="full",
            fold=fold,
            available_sample_ids=available,
            source_processed_manifest_path=source_manifest,
            project_root=root,
        )
        assert tuple(
            map(len, (selected.train_sample_ids, selected.validation_sample_ids, selected.test_sample_ids))
        ) == (360, 40, 100)
        held_out.extend(selected.test_sample_ids)
        assert not any(
            cv_payload["folds"][fold]["audit"]["cross_partition_evidence_session_overlap"].values()
        )
    assert len(held_out) == len(set(held_out)) == 500
    assert set(held_out) == available

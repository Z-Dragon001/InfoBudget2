from __future__ import annotations

import json
from pathlib import Path

import pytest

from infobudget.quality_router.candidate_corpus import build_candidate_fact_corpus


TIERS = {"small": "L_memories.json", "medium": "M_memories.json", "large": "H_memories.json"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, *, invalid_source: bool = False) -> tuple[Path, Path]:
    segments = tmp_path / "segments" / "samples" / "sample-1" / "segments.jsonl"
    segments.parent.mkdir(parents=True)
    segments.write_text(
        json.dumps(
            {
                "dataset_name": "dataset",
                "split": "full",
                "sample_id": "sample-1",
                "segment_id": "segment-1",
                "turn_ids": [1, 2],
                "source_content_hash": "segment-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    run_id = "run-1"
    run = runs / run_id
    models = {
        tier: {"model_id": f"model-{tier}", "model_name": f"model-{tier}"}
        for tier in TIERS
    }
    _write_json(
        run / "manifest.json",
        {
            "campaign_id": "campaign-1",
            "sample_id": "sample-1",
            "extraction_run_id": run_id,
            "status": "complete",
            "required_tiers": list(TIERS),
            "completed_tiers": list(TIERS),
            "qdrant_audit": {"reconciled": True},
            "models": models,
        },
    )
    for tier, filename in TIERS.items():
        _write_json(
            run / "human_readable" / "sample-1" / filename,
            {
                "metadata": {
                    "sample_id": "sample-1",
                    "collection_tier": tier,
                    "extraction_run_id": run_id,
                },
                "memories": [
                    {
                        "fact_id": f"fact-{tier}",
                        "fact_text": f"Fact from {tier}.",
                        "sample_id": "sample-1",
                        "segment_id": "segment-1",
                        "source_turn_ids": [99 if invalid_source and tier == "large" else 1],
                        "source_content_hash": "segment-hash",
                        "model_id": f"model-{tier}",
                        "memory_tier": tier,
                        "extraction_run_id": run_id,
                    }
                ],
            },
        )
    return runs, segments.parent.parent.parent


def test_build_candidate_fact_corpus_merges_all_models(tmp_path: Path) -> None:
    runs, segments = _fixture(tmp_path)
    output = tmp_path / "candidate_facts.jsonl"
    inventory_path = tmp_path / "inventory.json"
    inventory = build_candidate_fact_corpus(
        runs_root=runs,
        segments_path=segments,
        campaign_ids=["campaign-1"],
        output_path=output,
        inventory_path=inventory_path,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert {row["model_id"] for row in rows} == {
        "model-small",
        "model-medium",
        "model-large",
    }
    assert inventory["export_count"] == 3
    assert inventory["candidate_fact_count"] == 3


def test_build_candidate_fact_corpus_rejects_out_of_segment_sources(tmp_path: Path) -> None:
    runs, segments = _fixture(tmp_path, invalid_source=True)
    with pytest.raises(ValueError, match="out of segment"):
        build_candidate_fact_corpus(
            runs_root=runs,
            segments_path=segments,
            campaign_ids=["campaign-1"],
            output_path=tmp_path / "candidate_facts.jsonl",
            inventory_path=tmp_path / "inventory.json",
        )

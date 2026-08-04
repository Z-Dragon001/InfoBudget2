"""Extraction campaign consistency and aggregate quality gates."""

from __future__ import annotations

import json
from types import SimpleNamespace

from infobudget.rl_router.campaign import campaign_manifest_path, refresh_campaign
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.manifest import resolve_collection_namespace


def test_embedding_hash_is_part_of_collection_namespace() -> None:
    namespace = resolve_collection_namespace(
        {
            "collection_namespace": (
                "{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v2"
            )
        },
        dataset="longmemeval",
        split="full",
        segmentation_version="nsp_v1",
        embedding_hash="abcdef0123456789",
    )
    assert namespace == "longmemeval_full_nsp_v1_abcdef012345_fact_v2"


def test_campaign_requires_all_runs_and_aggregate_quality_to_pass(tmp_path) -> None:
    campaign_id = "campaign-test"
    bundle = SimpleNamespace(project=SimpleNamespace(root_dir=tmp_path))
    path = campaign_manifest_path(tmp_path, campaign_id)
    atomic_write_json(
        path,
        {
            "schema_version": "extraction_campaign_v1",
            "campaign_id": campaign_id,
            "campaign_scope_hash": "scope",
            "sample_count": 1,
            "expected_runs": {"sample-1": "run-1"},
            "required_tiers": ["small", "medium", "large"],
            "sample_segment_sha256": {"sample-1": "segments-hash"},
            "extraction_config": {
                "quality_gates": {
                    "max_empty_fact_segment_rate": 0.25,
                    "max_saturated_segment_rate": 0.10,
                    "max_repair_batch_rate": 0.20,
                    "max_failed_batch_rate": 0.0,
                }
            },
            "status": "planned",
        },
    )
    run_path = tmp_path / "outputs/rl_router/runs/run-1/manifest.json"
    run = {
        "campaign_id": campaign_id,
        "campaign_scope_hash": "scope",
        "status": "complete",
        "completed_tiers": ["small", "medium", "large"],
        "segments_jsonl_sha256": "segments-hash",
        "extraction_summary": {
            "quality_metrics": {
                "total_segment_results": 30,
                "empty_fact_segments": 3,
                "saturated_segments": 2,
                "total_batches": 10,
                "repair_batches": 1,
                "failed_batches": 0,
            }
        },
    }
    atomic_write_json(run_path, run)
    complete = refresh_campaign(bundle, campaign_id)
    assert complete["status"] == "complete"
    assert complete["quality_metrics"]["empty_fact_segment_rate"] == 0.1
    assert complete["completed_at"]

    run["extraction_summary"]["quality_metrics"]["empty_fact_segments"] = 10
    atomic_write_json(run_path, run)
    failed = refresh_campaign(bundle, campaign_id)
    assert failed["status"] == "quality_failed"
    assert "empty_fact_segment_rate" in failed["quality_violations"]

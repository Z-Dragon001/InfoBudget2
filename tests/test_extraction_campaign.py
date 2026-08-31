"""Extraction campaign consistency and aggregate quality gates."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from infobudget.rl_router.campaign import campaign_manifest_path, refresh_campaign
from infobudget.rl_router.campaign import initialize_campaign
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.manifest import resolve_collection_namespace


def test_project_and_embedding_hash_are_part_of_collection_namespace() -> None:
    namespace = resolve_collection_namespace(
        {
            "collection_namespace": (
                "{project_name}_{model_family}_{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v3"
            )
        },
        project_name="Info Budget 2",
        model_family="llama",
        dataset="longmemeval",
        split="full",
        segmentation_version="nsp_v1",
        embedding_hash="abcdef0123456789",
    )
    assert namespace == "info-budget-2_llama_longmemeval_full_nsp_v1_abcdef012345_fact_v3"
    qwen_namespace = resolve_collection_namespace(
        {
            "collection_namespace": (
                "{project_name}_{model_family}_{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v3"
            )
        },
        project_name="Info Budget 2",
        model_family="qwen",
        dataset="longmemeval",
        split="full",
        segmentation_version="nsp_v1",
        embedding_hash="abcdef0123456789",
    )
    assert qwen_namespace != namespace
    assert qwen_namespace.startswith("info-budget-2_qwen_")


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


def test_campaign_pins_only_the_selected_dataset_prompt(tmp_path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    locomo_prompt = prompt_root / "locomo.txt"
    longmemeval_prompt = prompt_root / "longmemeval.txt"
    locomo_prompt.write_text("locomo prompt v1", encoding="utf-8")
    longmemeval_prompt.write_text("longmemeval prompt v1", encoding="utf-8")
    segments = (
        tmp_path
        / "datasets"
        / "segmented"
        / "locomo"
        / "full"
        / "nsp_text_tiling"
        / "samples"
        / "sample-1"
        / "segments.jsonl"
    )
    segments.parent.mkdir(parents=True)
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    embedding_root = tmp_path / "embedding"
    embedding_root.mkdir()
    (embedding_root / "config.json").write_text("{}", encoding="utf-8")
    roles = {
        tier: SimpleNamespace(
            effective_model_name=f"model-{tier}", stable_model_id=f"model-{tier}"
        )
        for tier in ("small", "medium", "large")
    }
    paths = {"locomo": locomo_prompt, "longmemeval": longmemeval_prompt}
    versions = {"locomo": "locomo-v1", "longmemeval": "longmemeval-v1"}
    bundle = SimpleNamespace(
        project=SimpleNamespace(
            root_dir=tmp_path,
            models=roles,
            config=SimpleNamespace(project=SimpleNamespace(name="InfoBudget")),
        ),
        embeddings={"memory": {"local_path": "embedding"}},
        rl={
            "model_family": "qwen",
            "extraction": {"quality_gates": {}},
            "storage": {"collection_namespace": "test-{dataset}"},
        },
        fact_extraction_prompt_role=lambda dataset: f"fact_extraction_{dataset}",
        fact_extraction_prompt_version=lambda dataset: versions[dataset],
        fact_extraction_prompt_path=lambda dataset: paths[dataset],
    )

    created = initialize_campaign(
        bundle,
        campaign_id="locomo-v1",
        dataset_name="locomo",
        split="full",
        segmentation_method="nsp_text_tiling",
        run_prefix="locomo-v1",
    )
    assert created["fact_extraction_prompt_role"] == "fact_extraction_locomo"
    assert created["fact_extraction_prompt_version"] == "locomo-v1"
    assert created["prompt_sha256"] == hashlib.sha256(
        locomo_prompt.read_bytes()
    ).hexdigest()

    longmemeval_prompt.write_text("longmemeval prompt v2", encoding="utf-8")
    unchanged = initialize_campaign(
        bundle,
        campaign_id="locomo-v1",
        dataset_name="locomo",
        split="full",
        segmentation_method="nsp_text_tiling",
        run_prefix="locomo-v1",
    )
    assert unchanged["campaign_scope_hash"] == created["campaign_scope_hash"]

    locomo_prompt.write_text("locomo prompt v2", encoding="utf-8")
    with pytest.raises(ValueError, match="different immutable scope"):
        initialize_campaign(
            bundle,
            campaign_id="locomo-v1",
            dataset_name="locomo",
            split="full",
            segmentation_method="nsp_text_tiling",
            run_prefix="locomo-v1",
        )

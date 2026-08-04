"""Immutable full-dataset extraction campaign manifests and quality aggregation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infobudget.rl_router.config import RLConfigBundle
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.manifest import file_sha256, memory_embedding_hash
from infobudget.rl_router.schemas import TIERS


def campaign_manifest_path(root: Path, campaign_id: str) -> Path:
    return root / "outputs" / "rl_router" / "campaigns" / campaign_id / "manifest.json"


def initialize_campaign(
    bundle: RLConfigBundle,
    *,
    campaign_id: str,
    dataset_name: str,
    split: str,
    segmentation_method: str,
    run_prefix: str,
) -> dict[str, Any]:
    samples_root = (
        bundle.project.root_dir
        / "datasets"
        / "segmented"
        / dataset_name
        / split
        / segmentation_method
        / "samples"
    )
    segment_files = sorted(samples_root.glob("*/segments.jsonl"))
    if not segment_files:
        raise FileNotFoundError(f"campaign has no segmented samples under {samples_root}")
    sample_files = {path.parent.name: path.resolve() for path in segment_files}
    embedding_hash = memory_embedding_hash(bundle)
    scope = {
        "dataset_name": dataset_name,
        "split": split,
        "segmentation_method": segmentation_method,
        "required_tiers": list(TIERS),
        "sample_segment_sha256": {
            sample_id: file_sha256(path) for sample_id, path in sample_files.items()
        },
        "expected_runs": {
            sample_id: f"{run_prefix}_{segmentation_method}_{sample_id}"
            for sample_id in sample_files
        },
        "embedding_model_hash": embedding_hash,
        "models": {
            tier: bundle.project.models[tier].effective_model_name for tier in TIERS
        },
        "prompt_sha256": hashlib.sha256(
            bundle.prompt_path("fact_extraction").read_bytes()
        ).hexdigest(),
        "extraction_config": bundle.rl["extraction"],
        "collection_namespace_template": bundle.rl["storage"]["collection_namespace"],
    }
    serialized = json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    scope_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = campaign_manifest_path(bundle.project.root_dir, campaign_id)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("campaign_scope_hash") != scope_hash:
            raise ValueError(
                f"campaign {campaign_id} already exists with a different immutable scope"
            )
        return existing
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "extraction_campaign_v1",
        "campaign_id": campaign_id,
        "campaign_scope_hash": scope_hash,
        **scope,
        "sample_count": len(sample_files),
        "status": "planned",
        "created_at": now,
        "updated_at": now,
    }
    atomic_write_json(path, manifest)
    return manifest


def refresh_campaign(bundle: RLConfigBundle, campaign_id: str) -> dict[str, Any]:
    path = campaign_manifest_path(bundle.project.root_dir, campaign_id)
    if not path.is_file():
        raise FileNotFoundError(f"campaign manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "extraction_campaign_v1":
        raise ValueError("unsupported extraction campaign schema")
    expected_runs = dict(manifest["expected_runs"])
    required_tiers = set(manifest["required_tiers"])
    missing_runs: list[str] = []
    incomplete_runs: dict[str, Any] = {}
    totals = {
        "total_segment_results": 0,
        "empty_fact_segments": 0,
        "saturated_segments": 0,
        "total_batches": 0,
        "repair_batches": 0,
        "failed_batches": 0,
    }
    complete_samples = 0
    for sample_id, run_id in expected_runs.items():
        run_path = bundle.project.root_dir / "outputs" / "rl_router" / "runs" / run_id / "manifest.json"
        if not run_path.is_file():
            missing_runs.append(sample_id)
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        problems = []
        if run.get("campaign_id") != campaign_id:
            problems.append("campaign_id")
        if run.get("campaign_scope_hash") != manifest["campaign_scope_hash"]:
            problems.append("campaign_scope_hash")
        if run.get("status") != "complete":
            problems.append(f"status={run.get('status')}")
        if set(run.get("completed_tiers") or ()) != required_tiers:
            problems.append("completed_tiers")
        if run.get("segments_jsonl_sha256") != manifest["sample_segment_sha256"][sample_id]:
            problems.append("segments_jsonl_sha256")
        quality = (run.get("extraction_summary") or {}).get("quality_metrics") or {}
        if any(key not in quality for key in totals):
            problems.append("quality_metrics")
        if problems:
            incomplete_runs[sample_id] = problems
            continue
        complete_samples += 1
        for key in totals:
            totals[key] += int(quality.get(key, 0))

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    rates = {
        "empty_fact_segment_rate": rate(
            totals["empty_fact_segments"], totals["total_segment_results"]
        ),
        "saturated_segment_rate": rate(
            totals["saturated_segments"], totals["total_segment_results"]
        ),
        "repair_batch_rate": rate(totals["repair_batches"], totals["total_batches"]),
        "failed_batch_rate": rate(totals["failed_batches"], totals["total_batches"]),
    }
    gate_cfg = manifest["extraction_config"].get("quality_gates", {})
    thresholds = {
        "empty_fact_segment_rate": float(gate_cfg["max_empty_fact_segment_rate"]),
        "saturated_segment_rate": float(gate_cfg["max_saturated_segment_rate"]),
        "repair_batch_rate": float(gate_cfg["max_repair_batch_rate"]),
        "failed_batch_rate": float(gate_cfg["max_failed_batch_rate"]),
    }
    violations = {
        key: {"actual": rates[key], "maximum": maximum}
        for key, maximum in thresholds.items()
        if rates[key] > maximum
    }
    all_runs_complete = complete_samples == len(expected_runs)
    if not all_runs_complete:
        status = "incomplete"
    elif violations:
        status = "quality_failed"
    else:
        status = "complete"
    now = datetime.now(timezone.utc).isoformat()
    manifest.update(
        {
            "status": status,
            "complete_samples": complete_samples,
            "missing_samples": missing_runs,
            "incomplete_runs": incomplete_runs,
            "quality_metrics": {**totals, **rates},
            "quality_thresholds": thresholds,
            "quality_violations": violations,
            "updated_at": now,
        }
    )
    if status == "complete" and not manifest.get("completed_at"):
        manifest["completed_at"] = now
    atomic_write_json(path, manifest)
    return manifest


def load_complete_campaign(
    bundle: RLConfigBundle, campaign_id: str
) -> dict[str, Any]:
    manifest = refresh_campaign(bundle, campaign_id)
    validate_campaign_environment(bundle, manifest)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"extraction campaign {campaign_id} is not trainable: "
            f"status={manifest.get('status')}"
        )
    return manifest


def validate_campaign_environment(
    bundle: RLConfigBundle,
    manifest: dict[str, Any],
    *,
    precomputed_embedding_hash: str | None = None,
) -> None:
    actual = {
        "embedding_model_hash": (
            precomputed_embedding_hash or memory_embedding_hash(bundle)
        ),
        "models": {
            tier: bundle.project.models[tier].effective_model_name for tier in TIERS
        },
        "prompt_sha256": hashlib.sha256(
            bundle.prompt_path("fact_extraction").read_bytes()
        ).hexdigest(),
        "extraction_config": bundle.rl["extraction"],
        "collection_namespace_template": bundle.rl["storage"]["collection_namespace"],
    }
    mismatches = [key for key, value in actual.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "current configuration does not match the immutable extraction campaign: "
            + ", ".join(mismatches)
        )

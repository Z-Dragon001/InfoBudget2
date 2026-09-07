"""Build one audited, model-keyed candidate Fact corpus from extraction campaigns."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json


TIER_EXPORTS = {
    "small": "L_memories.json",
    "medium": "M_memories.json",
    "large": "H_memories.json",
}
TIER_ORDER = {tier: index for index, tier in enumerate(TIER_EXPORTS)}


def build_candidate_fact_corpus(
    *,
    runs_root: str | Path,
    segments_path: str | Path,
    campaign_ids: Iterable[str],
    output_path: str | Path,
    inventory_path: str | Path,
) -> dict[str, Any]:
    """Discover complete runs, validate their exports, and write one JSONL corpus."""
    root = Path(runs_root)
    campaigns = tuple(dict.fromkeys(str(item).strip() for item in campaign_ids if str(item).strip()))
    if not campaigns:
        raise ValueError("at least one campaign_id is required")
    if not root.is_dir():
        raise FileNotFoundError(root)

    segments = _load_segments(Path(segments_path))
    samples = sorted({sample_id for sample_id, _ in segments})
    manifests = _discover_manifests(root, campaigns)
    by_campaign_sample: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for manifest_path, manifest in manifests:
        key = (str(manifest.get("campaign_id") or ""), str(manifest.get("sample_id") or ""))
        if key in by_campaign_sample:
            raise ValueError(f"multiple runs found for campaign/sample: {key}")
        by_campaign_sample[key] = (manifest_path, manifest)

    missing = [
        (campaign_id, sample_id)
        for campaign_id in campaigns
        for sample_id in samples
        if (campaign_id, sample_id) not in by_campaign_sample
    ]
    if missing:
        raise ValueError(f"campaign runs are missing for samples: {missing[:20]}")

    rows: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()
    observed_model_campaign: dict[str, str] = {}

    for campaign_id in campaigns:
        for sample_id in samples:
            manifest_path, manifest = by_campaign_sample[(campaign_id, sample_id)]
            run_dir = manifest_path.parent
            run_id = str(manifest.get("extraction_run_id") or run_dir.name)
            _validate_manifest(manifest, run_id)
            required_tiers = tuple(manifest.get("required_tiers") or TIER_EXPORTS)
            unknown = sorted(set(required_tiers) - set(TIER_EXPORTS))
            if unknown:
                raise ValueError(f"unsupported tiers in {run_id}: {unknown}")

            manifest_models = manifest.get("models") or {}
            for tier in required_tiers:
                model_id = _manifest_model_id(manifest_models, tier, run_id)
                previous_campaign = observed_model_campaign.setdefault(model_id, campaign_id)
                if previous_campaign != campaign_id:
                    raise ValueError(
                        f"model_id {model_id!r} appears in multiple campaigns: "
                        f"{previous_campaign!r}, {campaign_id!r}"
                    )
                export_path = run_dir / "human_readable" / sample_id / TIER_EXPORTS[tier]
                fact_rows = _load_export(
                    export_path,
                    campaign_id=campaign_id,
                    run_id=run_id,
                    sample_id=sample_id,
                    tier=tier,
                    model_id=model_id,
                    segments=segments,
                    seen_fact_ids=seen_fact_ids,
                )
                rows.extend(fact_rows)
                exports.append(
                    {
                        "campaign_id": campaign_id,
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "tier": tier,
                        "model_id": model_id,
                        "source_file": str(export_path.resolve()),
                        "fact_count": len(fact_rows),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["sample_id"],
            row["segment_id"],
            row["model_id"],
            TIER_ORDER[row["memory_tier"]],
            int(row.get("fact_index", 0)),
            row["fact_id"],
        )
    )
    output = write_jsonl(output_path, rows)
    inventory = {
        "schema_version": "candidate_inventory_v2",
        "campaign_ids": list(campaigns),
        "dataset": sorted({value["dataset_name"] for value in segments.values()}),
        "split": sorted({value["split"] for value in segments.values()}),
        "sample_count": len(samples),
        "segment_count": len(segments),
        "run_count": len(campaigns) * len(samples),
        "export_count": len(exports),
        "candidate_fact_count": len(rows),
        "candidate_facts_sha256": file_sha256(output),
        "facts_by_tier": dict(sorted(Counter(row["memory_tier"] for row in rows).items())),
        "facts_by_model": dict(sorted(Counter(row["model_id"] for row in rows).items())),
        "models": sorted(observed_model_campaign),
        "exports": exports,
    }
    atomic_write_json(inventory_path, inventory)
    return inventory


def _load_segments(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if not {"dataset_name", "split", "sample_id", "segment_id", "turn_ids"}.issubset(row):
            continue
        key = (str(row["sample_id"]), str(row["segment_id"]))
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = {
            "dataset_name": str(row["dataset_name"]),
            "split": str(row["split"]),
            "turn_ids": {int(value) for value in row["turn_ids"]},
            "source_content_hash": str(row.get("source_content_hash") or ""),
        }
    if not result:
        raise ValueError(f"no segments found: {path}")
    return result


def _discover_manifests(
    runs_root: Path, campaign_ids: tuple[str, ...]
) -> list[tuple[Path, dict[str, Any]]]:
    selected = set(campaign_ids)
    result = []
    for path in sorted(runs_root.glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("campaign_id") or "") in selected:
            result.append((path, payload))
    if not result:
        raise ValueError(f"no run manifests found for campaigns: {campaign_ids}")
    return result


def _validate_manifest(manifest: dict[str, Any], run_id: str) -> None:
    required_tiers = set(manifest.get("required_tiers") or TIER_EXPORTS)
    completed_tiers = set(manifest.get("completed_tiers") or ())
    if manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_id}")
    if completed_tiers != required_tiers:
        raise ValueError(
            f"run tiers are incomplete: {run_id}; "
            f"required={sorted(required_tiers)}, completed={sorted(completed_tiers)}"
        )
    if (manifest.get("qdrant_audit") or {}).get("reconciled") is not True:
        raise ValueError(f"run Qdrant audit is not reconciled: {run_id}")


def _manifest_model_id(models: dict[str, Any], tier: str, run_id: str) -> str:
    model = models.get(tier) or {}
    model_id = str(model.get("model_id") or model.get("model_name") or "").strip()
    if not model_id:
        raise ValueError(f"manifest model_id is missing: run={run_id}, tier={tier}")
    return model_id


def _load_export(
    path: Path,
    *,
    campaign_id: str,
    run_id: str,
    sample_id: str,
    tier: str,
    model_id: str,
    segments: dict[tuple[str, str], dict[str, Any]],
    seen_fact_ids: set[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    memories = payload.get("memories")
    if not isinstance(memories, list):
        raise ValueError(f"memories must be a list: {path}")
    checks = {
        "sample_id": sample_id,
        "collection_tier": tier,
        "extraction_run_id": run_id,
    }
    for field, expected in checks.items():
        actual = str(metadata.get(field) or expected)
        if actual != expected:
            raise ValueError(f"export metadata mismatch for {field}: {path}")

    result = []
    for index, raw in enumerate(memories):
        if not isinstance(raw, dict):
            raise ValueError(f"memory row is not an object: {path}:{index}")
        row = dict(raw)
        fact_id = str(row.get("fact_id") or row.get("candidate_fact_id") or "").strip()
        fact_text = str(row.get("fact_text") or row.get("text") or row.get("fact") or "").strip()
        segment_id = str(row.get("segment_id") or "").strip()
        row_model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        row_tier = str(row.get("memory_tier") or tier).strip()
        sources = tuple(sorted({int(value) for value in row.get("source_turn_ids", row.get("source_ids", ())) }))
        if not fact_id or not fact_text or not segment_id or not sources:
            raise ValueError(f"candidate Fact is missing required fields: {path}:{index}")
        if fact_id in seen_fact_ids:
            raise ValueError(f"duplicate candidate fact_id: {fact_id}")
        if row_model_id != model_id or row_tier != tier:
            raise ValueError(f"candidate model/tier mismatch: {path}:{index}")
        key = (sample_id, segment_id)
        if key not in segments:
            raise ValueError(f"candidate references an unknown segment: {key}")
        segment = segments[key]
        invalid = sorted(set(sources) - segment["turn_ids"])
        if invalid:
            raise ValueError(f"candidate source_turn_ids are out of segment: {fact_id}: {invalid}")
        row_hash = str(row.get("source_content_hash") or row.get("segment_hash") or "")
        if row_hash and segment["source_content_hash"] and row_hash != segment["source_content_hash"]:
            raise ValueError(f"candidate segment hash mismatch: {fact_id}")
        seen_fact_ids.add(fact_id)
        row.update(
            {
                "dataset": segment["dataset_name"],
                "dataset_name": segment["dataset_name"],
                "split": segment["split"],
                "sample_id": sample_id,
                "segment_id": segment_id,
                "fact_id": fact_id,
                "candidate_fact_id": fact_id,
                "fact_text": fact_text,
                "text": fact_text,
                "model_id": model_id,
                "memory_tier": tier,
                "source_turn_ids": list(sources),
                "candidate_extraction_run_id": str(
                    row.get("candidate_extraction_run_id")
                    or row.get("extraction_run_id")
                    or run_id
                ),
                "campaign_id": campaign_id,
            }
        )
        result.append(row)
    return result

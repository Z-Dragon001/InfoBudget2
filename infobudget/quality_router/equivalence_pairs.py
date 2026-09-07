"""Create a frozen candidate/reference pair universe for semantic judging."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.quality_router.schemas import AtomicFact, FactSetKey
from infobudget.rl_router.ledger import atomic_write_json


def build_fact_equivalence_pairs(
    *,
    segments_path: str | Path,
    references_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write every source-overlapping pair in the same segment/model group."""
    segments = _load_segments(Path(segments_path))
    references, reference_hashes = _load_references(Path(references_path))
    candidates = _load_candidates(Path(candidates_path))

    missing_references = sorted(set(segments) - set(references))
    extra_references = sorted(set(references) - set(segments))
    if missing_references or extra_references:
        raise ValueError(
            "reference/segment key mismatch; "
            f"missing={missing_references[:10]}, extra={extra_references[:10]}"
        )
    unknown_candidate_keys = sorted(
        {key[:4] for key in candidates} - set(segments)
    )
    if unknown_candidate_keys:
        raise ValueError(
            f"candidates reference unknown segments: {unknown_candidate_keys[:10]}"
        )

    rows: list[dict[str, Any]] = []
    raw_pair_count = 0
    excluded_no_source_overlap = 0
    eligible_by_model: Counter[str] = Counter()
    excluded_by_model: Counter[str] = Counter()

    for candidate_key in sorted(candidates):
        segment_key = candidate_key[:4]
        model_id = candidate_key[4]
        reference_list = references[segment_key]
        for candidate, candidate_run_id in candidates[candidate_key]:
            for reference in reference_list:
                raw_pair_count += 1
                shared_sources = sorted(
                    set(candidate.source_turn_ids) & set(reference.source_turn_ids)
                )
                if not shared_sources:
                    excluded_no_source_overlap += 1
                    excluded_by_model[model_id] += 1
                    continue
                pair_id = _pair_id(segment_key, model_id, candidate, reference)
                rows.append(
                    {
                        "schema_version": "fact_equivalence_pair_v1",
                        "pair_id": pair_id,
                        "dataset": segment_key[0],
                        "dataset_name": segment_key[0],
                        "split": segment_key[1],
                        "sample_id": segment_key[2],
                        "segment_id": segment_key[3],
                        "model_id": model_id,
                        "candidate_fact_id": candidate.fact_id,
                        "candidate_fact_text": candidate.text,
                        "candidate_source_turn_ids": list(candidate.source_turn_ids),
                        "candidate_extraction_run_id": candidate_run_id,
                        "reference_fact_id": reference.fact_id,
                        "reference_fact_text": reference.text,
                        "reference_source_turn_ids": list(reference.source_turn_ids),
                        "shared_source_turn_ids": shared_sources,
                        "reference_set_hash": reference_hashes[segment_key],
                    }
                )
                eligible_by_model[model_id] += 1

    rows.sort(
        key=lambda row: (
            row["sample_id"],
            row["segment_id"],
            row["model_id"],
            row["candidate_fact_id"],
            row["reference_fact_id"],
        )
    )
    output = write_jsonl(output_path, rows)
    manifest = {
        "schema_version": "fact_equivalence_pair_manifest_v1",
        "segments_sha256": _path_digest(Path(segments_path)),
        "references_sha256": file_sha256(references_path),
        "candidates_sha256": file_sha256(candidates_path),
        "output_sha256": file_sha256(output),
        "segment_count": len(segments),
        "reference_fact_count": sum(len(items) for items in references.values()),
        "candidate_fact_count": sum(len(items) for items in candidates.values()),
        "model_count": len({key[4] for key in candidates}),
        "models": sorted({key[4] for key in candidates}),
        "raw_same_segment_pair_count": raw_pair_count,
        "eligible_pair_count": len(rows),
        "excluded_no_source_overlap_count": excluded_no_source_overlap,
        "eligible_pairs_by_model": dict(sorted(eligible_by_model.items())),
        "excluded_pairs_by_model": dict(sorted(excluded_by_model.items())),
        "pair_policy": {
            "same_dataset_split_sample_segment_required": True,
            "source_turn_overlap_required": True,
            "semantic_decision_included": False,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_segments(path: Path) -> dict[tuple[str, str, str, str], set[int]]:
    result: dict[tuple[str, str, str, str], set[int]] = {}
    for row in iter_jsonl(path):
        if not {"dataset_name", "split", "sample_id", "segment_id", "turn_ids"}.issubset(row):
            continue
        key = FactSetKey.from_dict(row).tuple()
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = {int(value) for value in row["turn_ids"]}
    if not result:
        raise ValueError(f"no segments found: {path}")
    return result


def _load_references(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str, str], list[AtomicFact]],
    dict[tuple[str, str, str, str], str],
]:
    result: dict[tuple[str, str, str, str], list[AtomicFact]] = {}
    hashes: dict[tuple[str, str, str, str], str] = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row).tuple()
        if key in result:
            raise ValueError(f"duplicate reference set: {key}")
        facts = [
            AtomicFact.from_dict(item, id_fields=("reference_fact_id", "fact_id"))
            for item in row.get("reference_facts", ())
        ]
        _require_unique_ids(facts, f"reference set {key}")
        result[key] = facts
        hashes[key] = str(row.get("reference_set_hash") or _fact_set_hash(facts))
    if not result:
        raise ValueError(f"no reference sets found: {path}")
    return result, hashes


def _load_candidates(
    path: Path,
) -> dict[tuple[str, str, str, str, str], list[tuple[AtomicFact, str]]]:
    result: dict[
        tuple[str, str, str, str, str], list[tuple[AtomicFact, str]]
    ] = defaultdict(list)
    seen_ids: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in iter_jsonl(path):
        segment_key = FactSetKey.from_dict(row).tuple()
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        if not model_id:
            raise ValueError("candidate row is missing model_id")
        key = (*segment_key, model_id)
        fact = AtomicFact.from_dict(row, id_fields=("fact_id", "candidate_fact_id"))
        if fact.fact_id in seen_ids[key]:
            raise ValueError(f"duplicate candidate fact_id in group {key}: {fact.fact_id}")
        seen_ids[key].add(fact.fact_id)
        run_id = str(
            row.get("candidate_extraction_run_id")
            or row.get("extraction_run_id")
            or ""
        ).strip()
        if not run_id:
            raise ValueError(f"candidate extraction run ID is missing: {fact.fact_id}")
        result[key].append((fact, run_id))
    if not result:
        raise ValueError(f"no candidate facts found: {path}")
    return dict(result)


def _pair_id(
    segment_key: tuple[str, str, str, str],
    model_id: str,
    candidate: AtomicFact,
    reference: AtomicFact,
) -> str:
    payload = [
        *segment_key,
        model_id,
        candidate.fact_id,
        reference.fact_id,
        candidate.text,
        list(candidate.source_turn_ids),
        reference.text,
        list(reference.source_turn_ids),
    ]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"eqp_{digest[:24]}"


def _fact_set_hash(facts: list[AtomicFact]) -> str:
    payload = [
        {"fact_id": fact.fact_id, "text": fact.text, "source_turn_ids": fact.source_turn_ids}
        for fact in sorted(facts, key=lambda item: item.fact_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _require_unique_ids(facts: list[AtomicFact], label: str) -> None:
    ids = [fact.fact_id for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate Fact IDs in {label}")

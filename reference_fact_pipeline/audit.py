"""Deterministic audit and human-review queue for frozen reference Facts."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from infobudget.quality_router.io import iter_jsonl, write_jsonl
from infobudget.rl_router.embedding import LocalSentenceEncoder
from infobudget.rl_router.ledger import atomic_write_json
from reference_fact_pipeline.io import load_topic_segments
from reference_fact_pipeline.pipeline import render_source_annotated_text


FACT_TYPES = {
    "identity",
    "state",
    "event",
    "plan",
    "preference",
    "goal",
    "relationship",
    "decision",
    "constraint",
    "health",
    "knowledge",
    "assistant_answer",
    "negative",
    "other",
}
STATE_STATUSES = {"current", "historical", "timeless", "unspecified"}
REQUIRED_ROW_FIELDS = {
    "schema_version",
    "dataset_name",
    "split",
    "sample_id",
    "session_id",
    "segment_id",
    "segment_order",
    "segmentation_method",
    "segmentation_version",
    "source_content_hash",
    "segment_turn_ids",
    "reference_set_hash",
    "reference_facts",
    "raw_proposal_count",
    "grounded_accept_count",
    "frozen_fact_count",
    "truncated_to_k",
    "run_id",
    "prompt_version",
    "config_hash",
}
REQUIRED_FACT_FIELDS = {
    "reference_fact_id",
    "fact_text",
    "text",
    "source_turn_ids",
    "fact_type",
    "state_status",
    "origin",
    "grounding_reason",
    "selection_rank",
}


def audit_reference_facts(
    *,
    references_path: str | Path,
    segments_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    raw_archive_dir: str | Path | None = None,
    ledger_path: str | Path | None = None,
    embedding_model_path: str | Path | None = None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dimension: int = 384,
    similarity_threshold: float = 0.90,
    random_sample_size: int = 120,
    max_facts_per_segment: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit one frozen reference collection without modifying it."""

    references_path = Path(references_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(references_path))
    segments = {item.segment_id: item for item in load_topic_segments(segments_path)}

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    seen_fact_ids: set[str] = set()
    fact_type_counts: Counter[str] = Counter()
    state_status_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    fact_counts: list[int] = []
    facts_for_sampling: list[dict[str, Any]] = []
    review_items: dict[str, dict[str, Any]] = {}
    facts_by_segment: dict[str, list[dict[str, Any]]] = {}

    for row_number, row in enumerate(rows, start=1):
        segment_id = str(row.get("segment_id") or "")
        missing_row_fields = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing_row_fields:
            _issue(
                errors,
                "missing_row_fields",
                segment_id=segment_id,
                row_number=row_number,
                fields=missing_row_fields,
            )
        if not segment_id:
            _issue(errors, "missing_segment_id", row_number=row_number)
            continue
        if segment_id in seen_segments:
            _issue(errors, "duplicate_reference_segment", segment_id=segment_id)
            continue
        seen_segments.add(segment_id)
        segment = segments.get(segment_id)
        if segment is None:
            _issue(errors, "missing_source_segment", segment_id=segment_id)
            continue

        _check_segment_metadata(row, segment, errors)
        facts = row.get("reference_facts")
        if not isinstance(facts, list):
            _issue(errors, "reference_facts_not_list", segment_id=segment_id)
            continue
        facts_by_segment[segment_id] = facts
        fact_counts.append(len(facts))

        if int(row.get("frozen_fact_count", -1)) != len(facts):
            _issue(
                errors,
                "frozen_fact_count_mismatch",
                segment_id=segment_id,
                declared=row.get("frozen_fact_count"),
                actual=len(facts),
            )
        if int(row.get("grounded_accept_count", -1)) < len(facts):
            _issue(errors, "accepted_count_less_than_frozen", segment_id=segment_id)
        if int(row.get("raw_proposal_count", -1)) < int(
            row.get("grounded_accept_count", 0)
        ):
            _issue(errors, "proposal_count_less_than_accepted", segment_id=segment_id)

        ranks = [fact.get("selection_rank") for fact in facts]
        if ranks != list(range(1, len(facts) + 1)):
            _issue(errors, "noncontiguous_selection_rank", segment_id=segment_id)

        if not facts:
            _add_review_item(
                review_items,
                category="zero_reference_set",
                segment=segment,
                facts=[],
                reason="该主题段没有冻结 Reference Fact，需要确认它确实只包含寒暄、空泛内容或不可记忆信息。",
            )
        if bool(row.get("truncated_to_k")) or len(facts) >= max_facts_per_segment:
            _add_review_item(
                review_items,
                category="fact_cap_reached",
                segment=segment,
                facts=facts,
                reason="Reference Fact 达到冻结上限，需要确认是否因截断遗漏重要事实。",
            )

        local_fact_ids: set[str] = set()
        local_keys: set[tuple[str, tuple[int, ...]]] = set()
        text_only_keys: set[str] = set()
        hash_payload: list[list[Any]] = []
        valid_turn_ids = set(segment.turn_ids)
        for fact in facts:
            fact_id = str(fact.get("reference_fact_id") or "")
            fact_text = str(fact.get("fact_text") or "")
            source_turn_ids = _integer_list(fact.get("source_turn_ids"))
            missing_fact_fields = sorted(REQUIRED_FACT_FIELDS - fact.keys())
            if missing_fact_fields:
                _issue(
                    errors,
                    "missing_fact_fields",
                    segment_id=segment_id,
                    fact_id=fact_id,
                    fields=missing_fact_fields,
                )
            if not fact_id or not fact_text or not source_turn_ids:
                _issue(
                    errors,
                    "empty_required_fact_value",
                    segment_id=segment_id,
                    fact_id=fact_id,
                )
            if fact.get("text") != fact_text:
                _issue(errors, "fact_text_alias_mismatch", segment_id=segment_id, fact_id=fact_id)
            if not str(fact.get("grounding_reason") or "").strip():
                _issue(errors, "empty_grounding_reason", segment_id=segment_id, fact_id=fact_id)
            if not set(source_turn_ids).issubset(valid_turn_ids):
                _issue(
                    errors,
                    "invalid_source_turn_ids",
                    segment_id=segment_id,
                    fact_id=fact_id,
                    source_turn_ids=source_turn_ids,
                )
            if fact.get("fact_type") not in FACT_TYPES:
                _issue(
                    errors,
                    "invalid_fact_type",
                    segment_id=segment_id,
                    fact_id=fact_id,
                    value=fact.get("fact_type"),
                )
            if fact.get("state_status") not in STATE_STATUSES:
                _issue(
                    errors,
                    "invalid_state_status",
                    segment_id=segment_id,
                    fact_id=fact_id,
                    value=fact.get("state_status"),
                )

            expected_fact_id = _reference_fact_id(segment_id, fact_text, source_turn_ids)
            if fact_id != expected_fact_id:
                _issue(
                    errors,
                    "reference_fact_id_hash_mismatch",
                    segment_id=segment_id,
                    fact_id=fact_id,
                    expected=expected_fact_id,
                )
            if fact_id in local_fact_ids:
                _issue(errors, "duplicate_fact_id_within_segment", segment_id=segment_id, fact_id=fact_id)
            if fact_id in seen_fact_ids:
                _issue(errors, "duplicate_fact_id_globally", segment_id=segment_id, fact_id=fact_id)
            local_fact_ids.add(fact_id)
            seen_fact_ids.add(fact_id)

            normalized = _normalize_fact(fact_text)
            exact_key = (normalized, tuple(source_turn_ids))
            if exact_key in local_keys:
                _issue(errors, "exact_duplicate_same_source", segment_id=segment_id, fact_id=fact_id)
            if normalized in text_only_keys:
                _issue(errors, "exact_duplicate_different_source", segment_id=segment_id, fact_id=fact_id)
            local_keys.add(exact_key)
            text_only_keys.add(normalized)

            fact_type_counts[str(fact.get("fact_type"))] += 1
            state_status_counts[str(fact.get("state_status"))] += 1
            origin_counts[str(fact.get("origin"))] += 1
            hash_payload.append([fact_id, fact_text, source_turn_ids])
            facts_for_sampling.append(
                {
                    "segment": segment,
                    "fact": fact,
                    "stratum": (
                        str(fact.get("fact_type")),
                        str(fact.get("state_status")),
                        str(fact.get("origin")),
                    ),
                }
            )

        expected_set_hash = hashlib.sha256(
            json.dumps(
                hash_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if row.get("reference_set_hash") != expected_set_hash:
            _issue(
                errors,
                "reference_set_hash_mismatch",
                segment_id=segment_id,
                expected=expected_set_hash,
            )

    missing_reference_segments = sorted(set(segments) - seen_segments)
    extra_reference_segments = sorted(seen_segments - set(segments))
    for segment_id in missing_reference_segments:
        _issue(errors, "missing_reference_row", segment_id=segment_id)
    for segment_id in extra_reference_segments:
        _issue(errors, "extra_reference_row", segment_id=segment_id)

    _check_manifest(manifest, rows, fact_counts, errors, warnings)
    raw_summary = _audit_raw_archive(raw_archive_dir, segments, review_items, warnings)
    ledger_summary = _audit_ledger(ledger_path, len(rows), errors)

    near_duplicate_summary = _near_duplicate_review(
        facts_by_segment=facts_by_segment,
        segments=segments,
        review_items=review_items,
        model_path=embedding_model_path,
        model_name=embedding_model_name,
        dimension=embedding_dimension,
        threshold=similarity_threshold,
    )
    _add_stratified_sample(
        facts_for_sampling,
        review_items,
        sample_size=random_sample_size,
        seed=seed,
    )

    status = "AUTOMATED_FAIL" if errors else "AUTOMATED_PASS_MANUAL_PENDING"
    summary = {
        "schema_version": "reference_fact_audit_v1",
        "status": status,
        "references_path": str(references_path.resolve()),
        "segments_path": str(Path(segments_path).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "segment_count": len(segments),
        "reference_row_count": len(rows),
        "fact_count": sum(fact_counts),
        "zero_fact_segment_count": sum(value == 0 for value in fact_counts),
        "fact_cap_segment_count": sum(value >= max_facts_per_segment for value in fact_counts),
        "fact_count_statistics": {
            "minimum": min(fact_counts) if fact_counts else 0,
            "maximum": max(fact_counts) if fact_counts else 0,
            "mean": float(np.mean(fact_counts)) if fact_counts else 0.0,
        },
        "fact_type_counts": dict(sorted(fact_type_counts.items())),
        "state_status_counts": dict(sorted(state_status_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "raw_archive": raw_summary,
        "ledger": ledger_summary,
        "near_duplicates": near_duplicate_summary,
        "manual_review_item_count": len(review_items),
        "manual_review_completion_required": True,
        "seed": seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "automated_audit.json", summary)
    write_jsonl(output_dir / "manual_review_queue.jsonl", review_items.values())
    (output_dir / "audit_report.md").write_text(
        _render_markdown(summary), encoding="utf-8"
    )
    return summary


def _check_segment_metadata(row: dict[str, Any], segment: Any, errors: list[dict[str, Any]]) -> None:
    expected = {
        "dataset_name": segment.dataset_name,
        "split": segment.split,
        "sample_id": segment.sample_id,
        "session_id": segment.session_id,
        "segment_id": segment.segment_id,
        "segment_order": segment.segment_order,
        "segmentation_method": segment.segmentation_method,
        "segmentation_version": segment.segmentation_version,
        "source_content_hash": segment.source_content_hash,
        "segment_turn_ids": list(segment.turn_ids),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            _issue(
                errors,
                "segment_metadata_mismatch",
                segment_id=segment.segment_id,
                field=field,
                actual=row.get(field),
                expected=value,
            )


def _check_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    fact_counts: list[int],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    if not bool(manifest.get("run_complete")):
        _issue(errors, "manifest_run_incomplete")
    if int(manifest.get("unresolved_failure_count", -1)) != 0:
        _issue(
            errors,
            "manifest_has_unresolved_failures",
            count=manifest.get("unresolved_failure_count"),
        )
    if int(manifest.get("segment_count", -1)) != len(rows):
        _issue(errors, "manifest_segment_count_mismatch")
    if int(manifest.get("fact_count", -1)) != sum(fact_counts):
        _issue(errors, "manifest_fact_count_mismatch")
    config_hashes = {str(row.get("config_hash")) for row in rows}
    run_ids = {str(row.get("run_id")) for row in rows}
    prompt_versions = {str(row.get("prompt_version")) for row in rows}
    if len(config_hashes) != 1 or manifest.get("config_hash") not in config_hashes:
        _issue(errors, "mixed_or_manifest_mismatched_config_hash")
    if len(run_ids) != 1 or manifest.get("run_id") not in run_ids:
        _issue(errors, "mixed_or_manifest_mismatched_run_id")
    if len(prompt_versions) != 1:
        _issue(errors, "mixed_prompt_versions", values=sorted(prompt_versions))
    if not bool(manifest.get("cost_complete")):
        _issue(
            warnings,
            "incomplete_cost_accounting",
            detail="不影响 Reference Fact 内容审计，但正式成本报告前需要补齐价格快照。",
        )


def _audit_raw_archive(
    raw_archive_dir: str | Path | None,
    segments: dict[str, Any],
    review_items: dict[str, dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if raw_archive_dir is None:
        return {"checked": False}
    root = Path(raw_archive_dir)
    if not root.is_dir():
        _issue(warnings, "raw_archive_missing", path=str(root))
        return {"checked": False, "missing": True}
    counts: Counter[str] = Counter()
    archived_segment_ids: set[str] = set()
    repaired_segment_ids: set[str] = set()
    invalid_json_files: list[str] = []
    for path in sorted(root.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json_files.append(str(path))
            continue
        stage = str(row.get("stage") or "unknown")
        segment_id = str(row.get("segment_id") or "")
        counts[stage] += 1
        if segment_id:
            archived_segment_ids.add(segment_id)
        if stage.endswith("_json_repair"):
            repaired_segment_ids.add(segment_id)
    for segment_id in sorted(repaired_segment_ids):
        segment = segments.get(segment_id)
        if segment is not None:
            _add_review_item(
                review_items,
                category="json_repair_used",
                segment=segment,
                facts=[],
                reason="该主题段的模型响应经过 JSON repair，需要确认修复只改变结构、未改变事实内容。",
            )
    if invalid_json_files:
        _issue(
            warnings,
            "invalid_raw_archive_json",
            count=len(invalid_json_files),
            examples=invalid_json_files[:10],
        )
    if set(segments) - archived_segment_ids:
        _issue(
            warnings,
            "segments_missing_raw_archive",
            count=len(set(segments) - archived_segment_ids),
        )
    return {
        "checked": True,
        "json_file_count": sum(counts.values()),
        "stage_counts": dict(sorted(counts.items())),
        "archived_segment_count": len(archived_segment_ids),
        "json_repair_segment_count": len(repaired_segment_ids),
        "invalid_json_file_count": len(invalid_json_files),
    }


def _audit_ledger(
    ledger_path: str | Path | None,
    expected_row_count: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    if ledger_path is None:
        return {"checked": False}
    path = Path(ledger_path).resolve()
    if not path.is_file():
        _issue(errors, "ledger_missing", path=str(path))
        return {"checked": False, "missing": True}
    uri = path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        quick_check = [row[0] for row in connection.execute("PRAGMA quick_check")]
        set_count = int(
            connection.execute("SELECT COUNT(*) FROM reference_fact_sets").fetchone()[0]
        )
        failure_count = int(
            connection.execute("SELECT COUNT(*) FROM reference_fact_failures").fetchone()[0]
        )
    if quick_check != ["ok"]:
        _issue(errors, "ledger_quick_check_failed", result=quick_check)
    if set_count != expected_row_count:
        _issue(
            errors,
            "ledger_reference_set_count_mismatch",
            expected=expected_row_count,
            actual=set_count,
        )
    if failure_count:
        _issue(errors, "ledger_contains_failures", count=failure_count)
    return {
        "checked": True,
        "quick_check": quick_check,
        "reference_fact_set_count": set_count,
        "failure_count": failure_count,
    }


def _near_duplicate_review(
    *,
    facts_by_segment: dict[str, list[dict[str, Any]]],
    segments: dict[str, Any],
    review_items: dict[str, dict[str, Any]],
    model_path: str | Path | None,
    model_name: str,
    dimension: int,
    threshold: float,
) -> dict[str, Any]:
    if model_path is None:
        return {"checked": False, "threshold": threshold}
    if not 0.0 < threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    texts: list[str] = []
    metadata: list[tuple[str, dict[str, Any]]] = []
    group_indices: dict[str, list[int]] = {}
    for segment_id, facts in facts_by_segment.items():
        indices: list[int] = []
        for fact in facts:
            indices.append(len(texts))
            texts.append(str(fact.get("fact_text") or ""))
            metadata.append((segment_id, fact))
        group_indices[segment_id] = indices
    encoder = LocalSentenceEncoder(
        model_name=model_name,
        local_path=model_path,
        dimension=dimension,
        normalize=True,
        max_length=256,
        long_text_strategy="truncate",
    )
    embeddings = encoder.encode(texts)
    hit_count = 0
    high_similarity_count = 0
    for segment_id, indices in group_indices.items():
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                similarity = float(embeddings[left_index] @ embeddings[right_index])
                if similarity < threshold:
                    continue
                hit_count += 1
                if similarity >= 0.95:
                    high_similarity_count += 1
                left_fact = metadata[left_index][1]
                right_fact = metadata[right_index][1]
                _add_review_item(
                    review_items,
                    category="near_duplicate_pair",
                    segment=segments[segment_id],
                    facts=[left_fact, right_fact],
                    reason=(
                        f"同一主题段内两条 Fact 的文本嵌入余弦相似度为 {similarity:.4f}；"
                        "需要人工判断是重复、包含关系，还是主体/方向不同的独立事实。"
                    ),
                    extra={"cosine_similarity": similarity},
                )
    return {
        "checked": True,
        "model_name": model_name,
        "model_path": str(Path(model_path).resolve()),
        "threshold": threshold,
        "pair_count": hit_count,
        "pair_count_ge_0_95": high_similarity_count,
    }


def _add_stratified_sample(
    facts: list[dict[str, Any]],
    review_items: dict[str, dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> None:
    if sample_size <= 0 or not facts:
        return
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in facts:
        grouped[item["stratum"]].append(item)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    strata = sorted(grouped)
    rng.shuffle(strata)
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < min(sample_size, len(facts)):
        made_progress = False
        for stratum in strata:
            values = grouped[stratum]
            if position < len(values):
                selected.append(values[position])
                made_progress = True
                if len(selected) >= min(sample_size, len(facts)):
                    break
        if not made_progress:
            break
        position += 1
    for item in selected:
        fact = item["fact"]
        _add_review_item(
            review_items,
            category="stratified_quality_sample",
            segment=item["segment"],
            facts=[fact],
            reason="分层随机人工审计：检查来源蕴含、原子性、记忆价值、类型以及时间状态。",
            extra={"stratum": list(item["stratum"])},
        )


def _add_review_item(
    review_items: dict[str, dict[str, Any]],
    *,
    category: str,
    segment: Any,
    facts: Iterable[dict[str, Any]],
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    fact_list = list(facts)
    fact_ids = [str(item.get("reference_fact_id") or "") for item in fact_list]
    identity = json.dumps(
        [category, segment.segment_id, fact_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    review_id = "review_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    review_items.setdefault(
        review_id,
        {
            "schema_version": "reference_fact_manual_review_v1",
            "review_id": review_id,
            "category": category,
            "dataset_name": segment.dataset_name,
            "split": segment.split,
            "sample_id": segment.sample_id,
            "session_id": segment.session_id,
            "segment_id": segment.segment_id,
            "segment_turn_ids": list(segment.turn_ids),
            "source_id_convention": "canonical_one_based",
            "source_id_rule": "SOURCE_TURN_ID = legacy dialogue index + 1",
            "source_id_mapping": [
                {
                    "legacy_dialogue_index": int(turn_id) - 1,
                    "source_turn_id": int(turn_id),
                }
                for turn_id in segment.turn_ids
            ],
            "segment_text": render_source_annotated_text(segment),
            "facts": fact_list,
            "review_reason": reason,
            "review_status": "pending",
            "review_decision": "",
            "reviewer_notes": "",
            "corrected_facts": [],
            **(extra or {}),
        },
    )


def _reference_fact_id(segment_id: str, fact_text: str, source_turn_ids: list[int]) -> str:
    payload = json.dumps(
        [segment_id, _normalize_fact(fact_text), source_turn_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "rf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _normalize_fact(text: str) -> str:
    return " ".join(str(text).casefold().strip().rstrip("。.!！?").split())


def _integer_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _issue(target: list[dict[str, Any]], code: str, **details: Any) -> None:
    target.append({"code": code, **details})


def _render_markdown(summary: dict[str, Any]) -> str:
    statistics = summary["fact_count_statistics"]
    near_duplicates = summary["near_duplicates"]
    lines = [
        "# Reference Fact 阶段一自动审计报告",
        "",
        f"- 自动状态：`{summary['status']}`",
        f"- Reference 行数：{summary['reference_row_count']}",
        f"- 来源 Segment 数：{summary['segment_count']}",
        f"- 冻结 Fact 数：{summary['fact_count']}",
        f"- 零 Fact Segment：{summary['zero_fact_segment_count']}",
        f"- 达到 Fact 上限的 Segment：{summary['fact_cap_segment_count']}",
        f"- 每段 Fact：min={statistics['minimum']}，max={statistics['maximum']}，mean={statistics['mean']:.3f}",
        f"- 自动错误：{summary['error_count']}",
        f"- 自动警告：{summary['warning_count']}",
        f"- 人工复核项：{summary['manual_review_item_count']}",
        "",
        "## 自动检查结论",
        "",
    ]
    if summary["errors"]:
        lines.append("自动审计未通过。必须修复以下错误后重新运行：")
        lines.append("")
        for issue in summary["errors"]:
            lines.append(f"- `{issue['code']}`：`{json.dumps(issue, ensure_ascii=False)}`")
    else:
        lines.append("结构、来源 ID、Fact ID、集合哈希、计数、manifest 和 SQLite ledger 自动检查通过。")
    lines.extend(["", "## 警告", ""])
    if summary["warnings"]:
        for issue in summary["warnings"]:
            lines.append(f"- `{issue['code']}`：`{json.dumps(issue, ensure_ascii=False)}`")
    else:
        lines.append("无自动警告。")
    lines.extend(
        [
            "",
            "## 语义近重复扫描",
            "",
            f"- 是否执行：{near_duplicates.get('checked', False)}",
            f"- 阈值：{near_duplicates.get('threshold')}",
            f"- 候选对数：{near_duplicates.get('pair_count', 0)}",
            f"- 相似度不低于 0.95：{near_duplicates.get('pair_count_ge_0_95', 0)}",
            "",
            "近重复结果只是人工复核候选。人物方向不同、邀请与接受、计划与完成等高相似文本可能仍然是独立事实，不能自动删除。",
            "",
            "## 阶段一通过门槛",
            "",
            "阶段一只有在以下条件全部满足后才能标记完成：",
            "",
            "1. `error_count == 0`；",
            "2. `manual_review_queue.jsonl` 中全部项目完成复核；",
            "3. 被判定需修改的 Fact 进入新版本 Reference 集合，不能原地静默修改冻结文件；",
            "4. 新版本重新计算 Fact ID、集合哈希和 manifest，并重新运行本审计；",
            "5. 最终审计报告和人工决策文件与 Reference 集合共同归档。",
            "",
            "当前状态仍为 `AUTOMATED_PASS_MANUAL_PENDING`，不等同于阶段一最终完成。",
            "",
        ]
    )
    return "\n".join(lines)

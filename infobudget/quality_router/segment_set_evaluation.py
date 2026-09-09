"""Resumable segment-level Candidate-set evaluation against reviewed Gold Facts."""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.api import ChatCompletionClient, ModelAPIError
from infobudget.rl_router.ledger import SqliteLedger, atomic_write_json
from infobudget.schemas import ModelSpec, PriceSpec
from infobudget.utils.text import count_tokens


PROMPT_VERSION = "segment_fact_set_judge_v3"
REPAIR_VERSION = "segment_fact_set_repair_v3_previous_response_and_policy"
SCHEMA_VERSION = "segment_fact_set_judgment_v2"
SEMANTIC_STATUSES = {
    "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"
}
COVERAGE_STATUSES = {"FULL", "PARTIAL", "NONE", "CONTRADICTED"}
TIME_STATUSES = {
    "PASS", "MISSING_REQUIRED_EXACT_TIME", "CONTRADICTED_TIME", "NOT_APPLICABLE"
}
MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)


def plan_segment_set_judging(
    *, segments_path: str | Path, references_path: str | Path,
    reference_manifest_path: str | Path, candidates_path: str | Path,
    candidate_inventory_path: str | Path, prompt_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec, anonymization_seed: int = 42,
) -> dict[str, Any]:
    _validate_reference_artifact(Path(references_path), Path(reference_manifest_path))
    model_ids = _validate_candidate_artifact(
        Path(candidates_path), Path(candidate_inventory_path)
    )
    tasks = _build_set_tasks(
        Path(segments_path), Path(references_path), Path(candidates_path),
        anonymization_seed, model_ids,
    )
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    prompts = [_render(prompt_text, task["model_input"]) for task in tasks]
    input_tokens = [count_tokens(prompt) for prompt in prompts]
    output_tokens = [_max_tokens(task, model_spec) for task in tasks]
    largest = max(input_tokens, default=0)
    if largest > model_spec.max_input_tokens:
        raise ValueError(
            f"largest prompt ({largest}) exceeds model input capacity "
            f"({model_spec.max_input_tokens})"
        )
    return {
        "schema_version": "segment_fact_set_judge_plan_v2",
        "paid_api_called": False,
        "segment_count": len(tasks),
        "logical_api_call_count": len(tasks),
        "judge_model": model_spec.effective_model_name,
        "prompt_version": PROMPT_VERSION,
        "repair_version": REPAIR_VERSION,
        "prompt_sha256": file_sha256(prompt_path),
        "segments_sha256": _path_digest(Path(segments_path)),
        "references_sha256": file_sha256(references_path),
        "candidates_sha256": file_sha256(candidates_path),
        "estimated_input_tokens": sum(input_tokens),
        "reserved_max_output_tokens": sum(output_tokens),
        "largest_estimated_input_tokens": largest,
        "estimated_upper_bound_input_cost": (
            sum(input_tokens) * price.official_price_in_per_1m / 1_000_000
        ),
        "estimated_upper_bound_output_cost": (
            sum(output_tokens) * price.official_price_out_per_1m / 1_000_000
        ),
        "currency": price.currency,
        "price_effective_date": price.price_effective_date,
    }


def run_segment_set_judging(
    *, segments_path: str | Path, references_path: str | Path,
    reference_manifest_path: str | Path, candidates_path: str | Path,
    candidate_inventory_path: str | Path, prompt_path: str | Path,
    output_dir: str | Path, output_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec, client: ChatCompletionClient,
    anonymization_seed: int = 42, max_segments: int | None = None,
    semantic_retries: int = 2,
) -> dict[str, Any]:
    segments_path = Path(segments_path)
    references_path = Path(references_path)
    reference_manifest_path = Path(reference_manifest_path)
    candidates_path = Path(candidates_path)
    candidate_inventory_path = Path(candidate_inventory_path)
    prompt_path = Path(prompt_path)
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    _validate_reference_artifact(references_path, reference_manifest_path)
    model_ids = _validate_candidate_artifact(
        candidates_path, candidate_inventory_path
    )
    tasks = _build_set_tasks(
        segments_path, references_path, candidates_path, anonymization_seed,
        model_ids,
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")
    identity = {
        "segments_sha256": _path_digest(segments_path),
        "references_sha256": file_sha256(references_path),
        "reference_manifest_sha256": file_sha256(reference_manifest_path),
        "candidates_sha256": file_sha256(candidates_path),
        "candidate_inventory_sha256": file_sha256(candidate_inventory_path),
        "prompt_sha256": file_sha256(prompt_path),
        "prompt_version": PROMPT_VERSION,
        "repair_version": REPAIR_VERSION,
        "judge_model": model_spec.effective_model_name,
        "anonymization_seed": int(anonymization_seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    _require_resume_identity(manifest_path, identity)
    ledger = SqliteLedger(
        output_dir / "judgments.sqlite3", "segment_judgments",
        key_fields=("segment_id",),
    )
    _recover_archives(
        output_dir=output_dir, tasks=tasks, ledger=ledger,
        identity=identity, model_spec=model_spec,
    )
    completed = {str(row["segment_id"]): row for row in ledger.read_all()}
    _write_current_artifacts(
        output_path=output_path, manifest_path=manifest_path,
        output_dir=output_dir, rows=list(completed.values()), identity=identity,
        total=len(tasks), price=price,
    )
    remaining = [task for task in tasks if task["segment_id"] not in completed]
    if max_segments is not None:
        remaining = _take_stratified(
            remaining, max(0, max_segments - len(completed))
        )

    for number, task in enumerate(remaining, start=1):
        prompt = _render(prompt_text, task["model_input"])
        segment_id = task["segment_id"]
        parsed = None
        response = None
        response_prompt = prompt
        validation_error = ""
        previous_response = ""
        for semantic_attempt in range(semantic_retries + 1):
            repair_prompt = prompt
            if validation_error:
                repair_prompt += _repair_instruction(
                    task, validation_error, previous_response
                )
            try:
                response = client.complete(
                    model_spec=model_spec, prompt=repair_prompt,
                    max_new_tokens=_max_tokens(task, model_spec), json_mode=True,
                )
            except ModelAPIError as exc:
                _archive_call(
                    output_dir, segment_id, repair_prompt, "transport_failed", "",
                    None, str(exc),
                )
                raise
            try:
                parsed = parse_segment_set_judgment(response.content, task)
            except ValueError as exc:
                validation_error = str(exc)
                previous_response = response.content
                _archive_call(
                    output_dir, segment_id, repair_prompt,
                    "invalid_semantic_response", response.content, response,
                    validation_error,
                )
                if semantic_attempt < semantic_retries:
                    print(
                        f"[repair] {segment_id}: invalid Judge JSON; "
                        f"retry {semantic_attempt + 1}/{semantic_retries}"
                    )
                    continue
                raise ValueError(
                    f"invalid set-Judge response for {segment_id} after "
                    f"{semantic_retries + 1} attempts: {validation_error}"
                ) from exc
            response_prompt = repair_prompt
            break
        if parsed is None or response is None:
            raise RuntimeError(f"set-Judge produced no result for {segment_id}")
        row = _judgment_row(
            parsed=parsed, task=task, identity=identity, model_spec=model_spec
        )
        ledger.append(row)
        _archive_call(
            output_dir, segment_id, response_prompt, "committed", response.content,
            response, "",
        )
        print(
            f"[set {number}] {segment_id}: "
            f"{len(parsed['candidate_set_results'])} sets committed"
        )

    rows = ledger.read_all()
    manifest = _write_current_artifacts(
        output_path=output_path, manifest_path=manifest_path,
        output_dir=output_dir, rows=rows, identity=identity,
        total=len(tasks), price=price,
    )
    return manifest


def parse_segment_set_judgment(
    content: str, task: dict[str, Any]
) -> dict[str, Any]:
    payload = _json_object(content)
    if payload.get("segment_id") != task["segment_id"]:
        raise ValueError("segment_id mismatch")
    results = payload.get("candidate_set_results")
    if not isinstance(results, list):
        raise ValueError("candidate_set_results must be a list")
    expected_sets = {
        item["set_id"]: item for item in task["model_input"]["candidate_sets"]
    }
    _canonicalize_structural_ids(
        results, "set_id", list(expected_sets), "Candidate set"
    )
    required_time = {
        str(fact["gold_fact_id"]): bool(fact["requires_exact_time"])
        for fact in task["model_input"]["gold_facts"]
    }
    clean_results = []
    validation_errors = []
    for result in results:
        set_id = str(result["set_id"])
        candidate_ids = {
            str(item["candidate_id"])
            for item in expected_sets[set_id]["facts"]
        }
        assessments = result.get("candidate_assessments")
        if not isinstance(assessments, list):
            raise ValueError(f"{set_id}: candidate_assessments must be a list")
        expected_candidate_order = [
            str(item["candidate_id"])
            for item in expected_sets[set_id]["facts"]
        ]
        _canonicalize_structural_ids(
            assessments, "candidate_id", expected_candidate_order,
            f"{set_id} Candidate",
        )
        status_by_candidate = {}
        for item in assessments:
            candidate_id = str(item["candidate_id"])
            status = item.get("semantic_status")
            if status not in SEMANTIC_STATUSES:
                validation_errors.append(
                    f"{set_id}/{candidate_id}: invalid semantic_status"
                )
            status_by_candidate[candidate_id] = status
            _collect_string_list_errors(
                validation_errors, item,
                ("supported_content", "unsupported_or_incorrect_content"),
                f"{set_id}/{candidate_id}",
            )

        gold = result.get("gold_fact_assessments")
        if not isinstance(gold, list):
            raise ValueError(f"{set_id}: gold_fact_assessments must be a list")
        _canonicalize_structural_ids(
            gold, "gold_fact_id", list(required_time), f"{set_id} Gold Fact"
        )
        for item in gold:
            gold_id = str(item["gold_fact_id"])
            coverage_status = item.get("coverage_status")
            if coverage_status not in COVERAGE_STATUSES:
                validation_errors.append(
                    f"{set_id}/{gold_id}: invalid coverage_status "
                    f"{coverage_status!r}; allowed={sorted(COVERAGE_STATUSES)}"
                )
            time_status = item.get("time_status")
            if time_status not in TIME_STATUSES:
                validation_errors.append(
                    f"{set_id}/{gold_id}: invalid time_status {time_status!r}; "
                    f"allowed={sorted(TIME_STATUSES)}"
                )
            elif required_time[gold_id] == (
                time_status == "NOT_APPLICABLE"
            ):
                validation_errors.append(
                    f"{set_id}/{gold_id}: time_status contradicts Gold time policy"
                )
            covering = item.get("covering_candidate_ids")
            covering_is_valid = not (
                not isinstance(covering, list)
                or not set(map(str, covering)).issubset(candidate_ids)
            )
            if not covering_is_valid:
                validation_errors.append(
                    f"{set_id}/{gold_id}: invalid covering_candidate_ids"
                )
            if (
                coverage_status in {"FULL", "PARTIAL"}
                and covering_is_valid and not covering
            ):
                validation_errors.append(
                    f"{set_id}/{gold_id}: covered Gold needs Candidate IDs"
                )
            if (
                time_status == "MISSING_REQUIRED_EXACT_TIME"
                and covering_is_valid
            ):
                invalid = [
                    candidate_id for candidate_id in map(str, covering)
                    if status_by_candidate[candidate_id] == "SUPPORTED"
                ]
                if invalid:
                    validation_errors.append(
                        f"{set_id}/{gold_id}: Candidates missing exact time "
                        f"cannot be SUPPORTED: {invalid}"
                    )
            _collect_string_list_errors(
                validation_errors, item,
                ("covered_content", "missing_or_incorrect_content"),
                f"{set_id}/{gold_id}",
            )
        clean_results.append(result)
    if validation_errors:
        details = "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(
            f"semantic validation failed with {len(validation_errors)} "
            f"error(s):\n{details}"
        )
    clean_results.sort(key=lambda item: item["set_id"])
    return {
        "segment_id": task["segment_id"],
        "candidate_set_results": clean_results,
    }


def gold_requires_exact_time(text: str) -> bool:
    """Conservative, deterministic trigger for the user's exact-time hard gate."""
    value = str(text).casefold()
    patterns = (
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
        rf"\b(?:{MONTHS})\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b",
        rf"\b\d{{1,2}}\s+(?:{MONTHS})\s+(?:19|20)\d{{2}}\b",
        rf"\b(?:{MONTHS})\s+(?:19|20)\d{{2}}\b",
        r"\b(?:19|20)\d{2}\b",
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
        r"\b\d+(?:\.\d+)?\s+(?:day|week|month|year)s?\b",
        r"\b(?:every\s+(?:day|week|month|year)|daily|weekly|monthly|yearly|"
        r"\d+\s+times?\s+(?:a|per)\s+(?:day|week|month|year))\b",
    )
    return any(re.search(pattern, value) for pattern in patterns)


def _canonicalize_structural_ids(
    items: Any, field: str, expected_order: list[str], label: str,
) -> None:
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{label} entries must be objects")
    returned = [str(item.get(field) or "") for item in items]
    if (
        len(returned) == len(expected_order)
        and len(set(returned)) == len(returned)
        and set(returned) == set(expected_order)
    ):
        return
    counts = Counter(returned)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    missing = sorted(set(expected_order) - set(returned))
    extra = sorted(set(returned) - set(expected_order))
    raise ValueError(
        f"{label} IDs invalid; expected_count={len(expected_order)}, "
        f"returned_count={len(items)}, missing={missing[:8]}, "
        f"extra={extra[:8]}, duplicated={duplicates[:8]}"
    )


def _repair_instruction(
    task: dict[str, Any], validation_error: str, previous_response: str
) -> str:
    candidate_sets = task["model_input"]["candidate_sets"]
    gold_facts = task["model_input"]["gold_facts"]
    requirements = {
        "segment_id": task["segment_id"],
        "set_ids": [str(item["set_id"]) for item in candidate_sets],
        "allowed_statuses": {
            "candidate_semantic_status": sorted(SEMANTIC_STATUSES),
            "gold_coverage_status": sorted(COVERAGE_STATUSES),
            "gold_time_status": sorted(TIME_STATUSES),
        },
        "candidate_ids_by_set": {
            str(item["set_id"]): [
                str(fact["candidate_id"]) for fact in item["facts"]
            ]
            for item in candidate_sets
        },
        "gold_fact_ids_required_in_every_set": [
            str(item["gold_fact_id"])
            for item in gold_facts
        ],
        "gold_time_policy_by_id": {
            str(item["gold_fact_id"]): {
                "requires_exact_time": bool(item["requires_exact_time"]),
                "allowed_time_statuses": (
                    ["PASS", "MISSING_REQUIRED_EXACT_TIME", "CONTRADICTED_TIME"]
                    if item["requires_exact_time"] else ["NOT_APPLICABLE"]
                ),
            }
            for item in gold_facts
        },
    }
    return (
        "\n\nREPAIR_INSTRUCTION:\n"
        "The previous response is included below as data. Repair that response; "
        "do not regenerate unrelated judgments from scratch. Return one complete "
        "replacement JSON object, not a patch or explanation.\n"
        f"VALIDATION_ERRORS:\n{validation_error}\n"
        "MANDATORY_REPAIR_RULES:\n"
        "- Copy every required set, Candidate, and Gold ID exactly once.\n"
        "- Candidate semantic_status and Gold coverage_status use different "
        "enums. Never use SUPPORTED or PARTIALLY_SUPPORTED as coverage_status.\n"
        "- If requires_exact_time is false, time_status must be NOT_APPLICABLE.\n"
        "- If requires_exact_time is true, time_status must never be "
        "NOT_APPLICABLE. Use MISSING_REQUIRED_EXACT_TIME when no Candidate "
        "preserves the required exact time, including when coverage is NONE; "
        "use CONTRADICTED_TIME for a conflicting time.\n"
        "- Equivalent date formats preserve exact time. For example, "
        "2023-05-07 and May 7, 2023 are the same date and must receive PASS.\n"
        "- Gold coverage_status measures non-temporal content. A Candidate can "
        "provide FULL non-temporal coverage while receiving "
        "MISSING_REQUIRED_EXACT_TIME; the time hard gate is represented "
        "separately by time_status.\n"
        "- Preserve valid judgments from the previous response and change every "
        "field identified by validation errors or mandatory rules.\n"
        "REQUIRED_STRUCTURE_AND_POLICY:\n"
        + json.dumps(requirements, ensure_ascii=False, sort_keys=True)
        + "\nPREVIOUS_INVALID_RESPONSE:\n"
        + previous_response
    )


def _build_set_tasks(
    segments_path: Path, references_path: Path, candidates_path: Path, seed: int,
    model_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    segments = _load_segments(segments_path)
    references = {
        tuple(_identity(row)): row for row in _load_references(references_path)
    }
    candidates: dict[
        tuple[str, str, str, str], dict[str, list[dict[str, str]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(candidates_path):
        key = tuple(_identity(row))
        model_id = str(
            row.get("model_id") or row.get("extractor_model") or ""
        ).strip()
        fact_id = str(
            row.get("fact_id") or row.get("candidate_fact_id") or ""
        ).strip()
        text = str(row.get("text") or row.get("fact_text") or "").strip()
        if not model_id or not fact_id or not text:
            raise ValueError("candidate row lacks model_id, fact_id, or text")
        candidates[key][model_id].append(
            {"candidate_id": fact_id, "text": text}
        )
    if set(references) != set(segments):
        missing = list(set(references) - set(segments))[:5]
        extra = list(set(segments) - set(references))[:5]
        raise ValueError(
            f"Segment/Gold scope mismatch; missing_segments={missing}, "
            f"segments_without_gold={extra}"
        )
    models = list(model_ids)
    unknown_models = sorted({
        model for groups in candidates.values() for model in groups
        if model not in model_ids
    })
    if unknown_models:
        raise ValueError(
            f"candidate corpus contains models absent from inventory: {unknown_models}"
        )
    unknown_segments = sorted(set(candidates) - set(references))
    if unknown_segments:
        raise ValueError(
            f"candidate corpus contains unknown segments: {unknown_segments[:5]}"
        )
    tasks = []
    for key, reference_row in sorted(references.items()):
        ordered_models = list(models)
        random.Random(f"{seed}:{key[-1]}").shuffle(ordered_models)
        model_by_set = {
            chr(ord("A") + index): model
            for index, model in enumerate(ordered_models)
        }
        candidate_sets = [
            {
                "set_id": set_id,
                "facts": sorted(
                    candidates.get(key, {}).get(model, ()),
                    key=lambda item: item["candidate_id"],
                ),
            }
            for set_id, model in model_by_set.items()
        ]
        gold_facts = [
            {
                "gold_fact_id": str(fact["reference_fact_id"]),
                "text": str(fact.get("text") or fact.get("fact_text")),
                "requires_exact_time": gold_requires_exact_time(
                    str(fact.get("text") or fact.get("fact_text"))
                ),
            }
            for fact in reference_row["reference_facts"]
        ]
        tasks.append(
            {
                "segment_id": key[-1], "dataset_name": key[0],
                "split": key[1], "sample_id": key[2],
                "reference_set_hash": str(
                    reference_row.get("reference_set_hash") or ""
                ),
                "model_by_set": model_by_set,
                "model_input": {
                    "segment_id": key[-1],
                    "segment_text": segments[key]["text"],
                    "gold_facts": gold_facts,
                    "candidate_sets": candidate_sets,
                },
            }
        )
    return tasks


def _load_references(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in iter_jsonl(path):
        if not isinstance(row.get("reference_facts"), list):
            continue
        key = tuple(_identity(row))
        if key in seen:
            raise ValueError(f"duplicate reference segment: {key}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"no reviewed Gold Fact sets found: {path}")
    return sorted(rows, key=_segment_sort_key)


def _load_segments(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result = {}
    for row in iter_jsonl(path):
        required = {"dataset_name", "split", "sample_id", "segment_id", "text"}
        if not required.issubset(row):
            continue
        key = tuple(_identity(row))
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"no segments found: {path}")
    return result


def _validate_reference_artifact(
    references_path: Path, manifest_path: Path
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "frozen_reference_manifest_v1":
        raise ValueError(
            "reference manifest must use frozen_reference_manifest_v1"
        )
    if manifest.get("run_complete") is not True:
        raise ValueError("reviewed Gold reference manifest is not complete")
    rows = _load_references(references_path)
    if int(manifest.get("segment_count", -1)) != len(rows):
        raise ValueError("reference manifest/segment count mismatch")
    fact_count = sum(len(row["reference_facts"]) for row in rows)
    if int(manifest.get("fact_count", -1)) != fact_count:
        raise ValueError("reference manifest/Gold Fact count mismatch")
    review = manifest.get("manual_review") or {}
    if review.get("complete") is not True or review.get("final_audit_complete") is not True:
        raise ValueError("reviewed Gold manual review/final audit is incomplete")


def _validate_candidate_artifact(
    candidates_path: Path, inventory_path: Path
) -> tuple[str, ...]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema_version") not in {
        "candidate_inventory_v1", "candidate_inventory_v2"
    }:
        raise ValueError("candidate inventory has an unsupported schema_version")
    if inventory.get("candidate_facts_sha256") != file_sha256(candidates_path):
        raise ValueError("candidate inventory/corpus hash mismatch")
    models = tuple(sorted(str(model).strip() for model in inventory.get("models", ())))
    if not models or any(not model for model in models) or len(models) != len(set(models)):
        raise ValueError("candidate inventory models are missing or duplicated")
    if int(inventory.get("candidate_fact_count", -1)) != sum(
        1 for _ in iter_jsonl(candidates_path)
    ):
        raise ValueError("candidate inventory/Fact count mismatch")
    return models


def _judgment_row(
    *, parsed: dict[str, Any], task: dict[str, Any],
    identity: dict[str, Any], model_spec: ModelSpec,
    recovered_from_raw_call: str = "",
) -> dict[str, Any]:
    for result in parsed["candidate_set_results"]:
        result["model_id"] = task["model_by_set"][result["set_id"]]
    return {
        **parsed,
        "schema_version": SCHEMA_VERSION,
        "dataset_name": task["dataset_name"],
        "dataset": task["dataset_name"],
        "split": task["split"], "sample_id": task["sample_id"],
        "reference_set_hash": task["reference_set_hash"],
        "set_model_map": task["model_by_set"],
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": identity["prompt_sha256"],
        "judge_model": model_spec.effective_model_name,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "recovered_from_raw_call": recovered_from_raw_call,
    }


def _recover_archives(
    *, output_dir: Path, tasks: list[dict[str, Any]], ledger: SqliteLedger,
    identity: dict[str, Any], model_spec: ModelSpec,
) -> int:
    raw_dir = output_dir / "raw_calls"
    if not raw_dir.is_dir():
        return 0
    by_segment = {task["segment_id"]: task for task in tasks}
    completed = {str(row["segment_id"]) for row in ledger.read_all()}
    recovered = 0
    for path in sorted(raw_dir.glob("*.json"), reverse=True):
        archive = json.loads(path.read_text(encoding="utf-8"))
        segment_id = str(archive.get("segment_id") or "")
        if (
            segment_id in completed or segment_id not in by_segment
            or archive.get("status") != "invalid_semantic_response"
            or not str(archive.get("response_content") or "").strip()
        ):
            continue
        try:
            parsed = parse_segment_set_judgment(
                str(archive["response_content"]), by_segment[segment_id]
            )
        except (ValueError, json.JSONDecodeError):
            continue
        row = _judgment_row(
            parsed=parsed, task=by_segment[segment_id], identity=identity,
            model_spec=model_spec, recovered_from_raw_call=path.name,
        )
        if ledger.append(row):
            completed.add(segment_id)
            recovered += 1
            print(
                f"[recover] {segment_id}: committed archived response "
                "without an API call"
            )
    return recovered


def _write_current_artifacts(
    *, output_path: Path, manifest_path: Path, output_dir: Path,
    rows: list[dict[str, Any]], identity: dict[str, Any], total: int,
    price: PriceSpec,
) -> dict[str, Any]:
    rows.sort(key=_segment_sort_key)
    write_jsonl(output_path, rows)
    usage = _usage_totals(output_dir)
    complete = len(rows) == total
    manifest = {
        "schema_version": "segment_fact_set_judge_manifest_v2",
        **identity,
        "status": "complete" if complete else "incomplete",
        "run_complete": complete,
        "segment_count": total,
        "completed_segment_count": len(rows),
        "remaining_segment_count": total - len(rows),
        **usage,
        "input_cost": (
            usage["input_tokens"] * price.official_price_in_per_1m / 1_000_000
        ),
        "output_cost": (
            usage["output_tokens"] * price.official_price_out_per_1m / 1_000_000
        ),
        "total_cost": (
            usage["input_tokens"] * price.official_price_in_per_1m / 1_000_000
            + usage["output_tokens"] * price.official_price_out_per_1m / 1_000_000
        ),
        "currency": price.currency,
        "price_effective_date": price.price_effective_date,
        "output": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "evaluation_policy": {
            "unit": "reviewed_gold_fact_against_anonymous_candidate_set",
            "coverage_labels": ["FULL", "PARTIAL", "NONE", "CONTRADICTED"],
            "partial_credit": 0.5,
            "source_provenance_evaluated": False,
            "redundancy_evaluated": False,
            "gold_decomposition_used": False,
            "exact_gold_time_is_hard_gate": True,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset_name") or row.get("dataset") or ""),
        str(row.get("split") or ""), str(row.get("sample_id") or ""),
        str(row.get("segment_id") or ""),
    )


def _render(prompt_text: str, payload: dict[str, Any]) -> str:
    return (
        prompt_text.rstrip() + "\n\nINPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response root must be an object")
    return value


def _collect_string_list_errors(
    errors: list[str], item: dict[str, Any], fields: Iterable[str], context: str,
) -> None:
    for field in fields:
        value = item.get(field)
        if (
            not isinstance(value, list)
            or any(not isinstance(element, str) for element in value)
        ):
            errors.append(f"{context}: {field} must be a list of strings")


def _max_tokens(task: dict[str, Any], model: ModelSpec) -> int:
    candidates = sum(
        len(item["facts"])
        for item in task["model_input"]["candidate_sets"]
    )
    gold = len(task["model_input"]["gold_facts"])
    return min(model.max_output_tokens, max(2048, 260 * (candidates + gold)))


def _archive_call(
    output_dir: Path, segment_id: str, prompt: str, status: str,
    content: str, response: Any, error: str,
) -> None:
    raw = output_dir / "raw_calls"
    raw.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = raw / f"{stamp}_{uuid.uuid4().hex[:8]}.json"
    usage = None if response is None else {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "retry_count": response.retry_count,
        "provider_request_id": response.provider_request_id,
        "finish_reason": response.finish_reason,
    }
    atomic_write_json(
        path,
        {
            "schema_version": "segment_set_llm_call_v2",
            "segment_id": segment_id, "status": status,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_content": content, "usage": usage, "error": error,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _usage_totals(output_dir: Path) -> dict[str, int]:
    totals: Counter[str] = Counter(
        {"logical_api_call_count": 0, "input_tokens": 0, "output_tokens": 0}
    )
    raw = output_dir / "raw_calls"
    for path in raw.glob("*.json") if raw.is_dir() else ():
        row = json.loads(path.read_text(encoding="utf-8"))
        usage = row.get("usage") or {}
        totals["logical_api_call_count"] += 1
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
    return {
        "logical_api_call_count": totals["logical_api_call_count"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
    }


def _require_resume_identity(
    path: Path, identity: dict[str, Any]
) -> None:
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: (existing.get(key), value)
        for key, value in identity.items() if existing.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume identity mismatch: {mismatches}")


def _path_digest(path: Path) -> str:
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL files found: {path}")
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(path).as_posix() if path.is_dir() else file.name
        digest.update(relative.encode("utf-8"))
        digest.update(file_sha256(file).encode("ascii"))
    return digest.hexdigest()


def _take_stratified(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id") or "")].append(row)
    selected: list[dict[str, Any]] = []
    offsets = {sample_id: 0 for sample_id in grouped}
    while len(selected) < limit:
        progressed = False
        for sample_id in sorted(grouped):
            offset = offsets[sample_id]
            if offset >= len(grouped[sample_id]):
                continue
            selected.append(grouped[sample_id][offset])
            offsets[sample_id] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _segment_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    order = row.get("segment_order", row.get("segment_index", 0))
    return (
        str(row.get("sample_id", "")), int(order or 0),
        str(row.get("segment_id", "")),
    )

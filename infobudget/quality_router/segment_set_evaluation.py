"""Resumable Gold normalization and segment-level Candidate-set evaluation."""

from __future__ import annotations

import hashlib
import json
import random
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


GOLD_PROMPT_VERSION = "gold_evaluation_unit_builder_v1"
GOLD_SCHEMA_VERSION = "gold_evaluation_units_v1"
SET_PROMPT_VERSION = "segment_fact_set_judge_v1"
SET_SCHEMA_VERSION = "segment_fact_set_judgment_v1"

SEMANTIC_STATUSES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
}
CONTENT_STATUSES = {"COVERED", "PARTIAL", "NOT_COVERED", "CONTRADICTED"}
TIME_STATUSES = {
    "PASS",
    "MISSING_REQUIRED_EXACT_TIME",
    "CONTRADICTED_TIME",
    "NOT_APPLICABLE",
}
TIME_RESOLUTIONS = {"day", "month", "year", "time", "duration", "frequency", None}
TIME_RESOLUTION_ALIASES = {
    "date": "day",
    "calendar_day": "day",
    "clock": "time",
    "clock_time": "time",
}
NULL_LIKE_TIME_VALUES = {"", "null", "none", "n/a", "na", "not_applicable", "not applicable"}
NON_EXACT_TIME_RESOLUTIONS = {"current", "present", "relative", "vague", "unspecified", "state"}
NON_EXACT_TIME_SURFACES = {
    "currently", "current", "now", "today", "yesterday", "recently", "lately",
    "previously", "last day", "last week", "last month", "last year",
}


def plan_gold_evaluation_units(
    *, references_path: str | Path, prompt_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec,
) -> dict[str, Any]:
    references = _load_references(Path(references_path))
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    prompts = [_render(prompt_text, _gold_request(row)) for row in references]
    return _plan(
        prompts=prompts,
        item_count=len(references),
        output_reservations=[_gold_max_tokens(row, model_spec) for row in references],
        model_spec=model_spec,
        price=price,
        schema_version="gold_evaluation_unit_plan_v1",
        prompt_version=GOLD_PROMPT_VERSION,
        input_hash=file_sha256(references_path),
        input_hash_name="references_sha256",
        prompt_path=prompt_path,
    )


def run_gold_evaluation_units(
    *, references_path: str | Path, prompt_path: str | Path,
    output_dir: str | Path, output_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec, client: ChatCompletionClient,
    max_segments: int | None = None,
) -> dict[str, Any]:
    references_path = Path(references_path)
    prompt_path = Path(prompt_path)
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    references = _load_references(references_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    identity = {
        "references_sha256": file_sha256(references_path),
        "prompt_sha256": file_sha256(prompt_path),
        "prompt_version": GOLD_PROMPT_VERSION,
        "judge_model": model_spec.effective_model_name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    _require_resume_identity(manifest_path, identity)
    ledger = SqliteLedger(output_dir / "evaluation.sqlite3", "gold_units", key_fields=("segment_id",))
    recovered_count = _recover_gold_archives(
        output_dir=output_dir,
        references=references,
        ledger=ledger,
        identity=identity,
        model_spec=model_spec,
    )
    completed = {str(row["segment_id"]): row for row in ledger.read_all()}
    write_jsonl(output_path, sorted(completed.values(), key=_segment_sort_key))
    atomic_write_json(
        manifest_path,
        _manifest(
            identity=identity,
            schema_version="gold_evaluation_unit_manifest_v1",
            total=len(references), completed=len(completed), output_path=output_path,
            output_dir=output_dir, price=price,
        ),
    )
    remaining = [row for row in references if str(row["segment_id"]) not in completed]
    if max_segments is not None:
        remaining = _take_stratified(remaining, max(0, max_segments - recovered_count))

    for number, reference_row in enumerate(remaining, start=1):
        request = _gold_request(reference_row)
        prompt = _render(prompt_text, request)
        segment_id = str(reference_row["segment_id"])
        try:
            response = client.complete(
                model_spec=model_spec,
                prompt=prompt,
                max_new_tokens=_gold_max_tokens(reference_row, model_spec),
                json_mode=True,
            )
        except ModelAPIError as exc:
            _archive_call(output_dir, segment_id, prompt, "transport_failed", "", None, str(exc))
            raise
        try:
            parsed = parse_gold_evaluation_units(response.content, reference_row)
        except ValueError as exc:
            _archive_call(output_dir, segment_id, prompt, "invalid_semantic_response", response.content, response, str(exc))
            raise ValueError(f"invalid Gold-unit response for {segment_id}: {exc}") from exc
        row = _gold_output_row(
            parsed=parsed,
            reference_row=reference_row,
            identity=identity,
            model_spec=model_spec,
        )
        ledger.append(row)
        _archive_call(output_dir, segment_id, prompt, "committed", response.content, response, "")
        print(f"[gold {number}] {segment_id}: committed")

    rows = sorted(ledger.read_all(), key=_segment_sort_key)
    write_jsonl(output_path, rows)
    manifest = _manifest(
        identity=identity,
        schema_version="gold_evaluation_unit_manifest_v1",
        total=len(references), completed=len(rows), output_path=output_path,
        output_dir=output_dir, price=price,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def plan_segment_set_judging(
    *, segments_path: str | Path, gold_units_path: str | Path,
    gold_units_manifest_path: str | Path, candidates_path: str | Path,
    candidate_inventory_path: str | Path, prompt_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec, anonymization_seed: int = 42,
) -> dict[str, Any]:
    _validate_gold_units_artifact(Path(gold_units_path), Path(gold_units_manifest_path))
    _validate_candidate_artifact(Path(candidates_path), Path(candidate_inventory_path))
    tasks = _build_set_tasks(
        Path(segments_path), Path(gold_units_path), Path(candidates_path), anonymization_seed
    )
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    prompts = [_render(prompt_text, task["model_input"]) for task in tasks]
    return _plan(
        prompts=prompts,
        item_count=len(tasks),
        output_reservations=[_set_max_tokens(task, model_spec) for task in tasks],
        model_spec=model_spec,
        price=price,
        schema_version="segment_fact_set_judge_plan_v1",
        prompt_version=SET_PROMPT_VERSION,
        input_hash=_combined_hash(segments_path, gold_units_path, candidates_path),
        input_hash_name="inputs_sha256",
        prompt_path=prompt_path,
    )


def run_segment_set_judging(
    *, segments_path: str | Path, gold_units_path: str | Path,
    gold_units_manifest_path: str | Path, candidates_path: str | Path,
    candidate_inventory_path: str | Path, prompt_path: str | Path,
    output_dir: str | Path, output_path: str | Path,
    model_spec: ModelSpec, price: PriceSpec, client: ChatCompletionClient,
    anonymization_seed: int = 42, max_segments: int | None = None,
) -> dict[str, Any]:
    _validate_gold_units_artifact(Path(gold_units_path), Path(gold_units_manifest_path))
    _validate_candidate_artifact(Path(candidates_path), Path(candidate_inventory_path))
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    prompt_path = Path(prompt_path)
    tasks = _build_set_tasks(
        Path(segments_path), Path(gold_units_path), Path(candidates_path), anonymization_seed
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")
    identity = {
        "segments_sha256": _path_digest(Path(segments_path)),
        "gold_units_sha256": file_sha256(gold_units_path),
        "gold_units_manifest_sha256": file_sha256(gold_units_manifest_path),
        "candidates_sha256": file_sha256(candidates_path),
        "candidate_inventory_sha256": file_sha256(candidate_inventory_path),
        "prompt_sha256": file_sha256(prompt_path),
        "prompt_version": SET_PROMPT_VERSION,
        "judge_model": model_spec.effective_model_name,
        "anonymization_seed": int(anonymization_seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    _require_resume_identity(manifest_path, identity)
    ledger = SqliteLedger(output_dir / "judgments.sqlite3", "segment_judgments", key_fields=("segment_id",))
    completed = {str(row["segment_id"]): row for row in ledger.read_all()}
    write_jsonl(output_path, sorted(completed.values(), key=_segment_sort_key))
    initial_manifest = _manifest(
        identity=identity,
        schema_version="segment_fact_set_judge_manifest_v1",
        total=len(tasks), completed=len(completed), output_path=output_path,
        output_dir=output_dir, price=price,
    )
    initial_manifest["evaluation_policy"] = {
        "unit": "one_segment_with_anonymous_candidate_sets",
        "source_provenance_evaluated": False,
        "redundancy_evaluated": False,
        "exact_gold_time_is_hard_gate": True,
    }
    atomic_write_json(manifest_path, initial_manifest)
    remaining = [task for task in tasks if task["segment_id"] not in completed]
    if max_segments is not None:
        remaining = _take_stratified(remaining, max_segments)

    for number, task in enumerate(remaining, start=1):
        prompt = _render(prompt_text, task["model_input"])
        segment_id = task["segment_id"]
        try:
            response = client.complete(
                model_spec=model_spec,
                prompt=prompt,
                max_new_tokens=_set_max_tokens(task, model_spec),
                json_mode=True,
            )
        except ModelAPIError as exc:
            _archive_call(output_dir, segment_id, prompt, "transport_failed", "", None, str(exc))
            raise
        try:
            parsed = parse_segment_set_judgment(response.content, task)
        except ValueError as exc:
            _archive_call(output_dir, segment_id, prompt, "invalid_semantic_response", response.content, response, str(exc))
            raise ValueError(f"invalid set-Judge response for {segment_id}: {exc}") from exc
        model_by_set = task["model_by_set"]
        for result in parsed["candidate_set_results"]:
            result["model_id"] = model_by_set[result["set_id"]]
        row = {
            **parsed,
            "schema_version": SET_SCHEMA_VERSION,
            "dataset_name": task["dataset_name"],
            "dataset": task["dataset_name"],
            "split": task["split"],
            "sample_id": task["sample_id"],
            "set_model_map": model_by_set,
            "prompt_version": SET_PROMPT_VERSION,
            "prompt_sha256": identity["prompt_sha256"],
            "judge_model": model_spec.effective_model_name,
            "judged_at": datetime.now(timezone.utc).isoformat(),
        }
        ledger.append(row)
        _archive_call(output_dir, segment_id, prompt, "committed", response.content, response, "")
        print(f"[set {number}] {segment_id}: {len(parsed['candidate_set_results'])} sets committed")

    rows = sorted(ledger.read_all(), key=_segment_sort_key)
    write_jsonl(output_path, rows)
    manifest = _manifest(
        identity=identity,
        schema_version="segment_fact_set_judge_manifest_v1",
        total=len(tasks), completed=len(rows), output_path=output_path,
        output_dir=output_dir, price=price,
    )
    manifest["evaluation_policy"] = {
        "unit": "one_segment_with_anonymous_candidate_sets",
        "source_provenance_evaluated": False,
        "redundancy_evaluated": False,
        "exact_gold_time_is_hard_gate": True,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_gold_evaluation_units(content: str, reference_row: dict[str, Any]) -> dict[str, Any]:
    payload = _json_object(content)
    segment_id = str(reference_row["segment_id"])
    if payload.get("segment_id") != segment_id:
        raise ValueError("segment_id mismatch")
    supplied = {str(item["reference_fact_id"]): str(item.get("text") or item.get("fact_text")) for item in reference_row["reference_facts"]}
    facts = payload.get("gold_facts")
    if not isinstance(facts, list):
        raise ValueError("gold_facts must be a list")
    returned_ids = [str(item.get("gold_fact_id") or "") for item in facts if isinstance(item, dict)]
    if len(facts) != len(returned_ids) or set(returned_ids) != set(supplied) or len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Gold Fact IDs are missing, extra, or duplicated")
    claim_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for fact in facts:
        fact_id = str(fact["gold_fact_id"])
        if fact.get("original_text") != supplied[fact_id]:
            raise ValueError(f"{fact_id}: original_text changed")
        units = fact.get("claim_units")
        if not isinstance(units, list) or not units:
            raise ValueError(f"{fact_id}: claim_units must be non-empty")
        clean_units = []
        for unit in units:
            if not isinstance(unit, dict):
                raise ValueError(f"{fact_id}: claim unit must be an object")
            claim_id = str(unit.get("claim_id") or "")
            if not claim_id.startswith(f"{fact_id}:C") or claim_id in claim_ids:
                raise ValueError(f"invalid or duplicate claim_id: {claim_id!r}")
            claim_ids.add(claim_id)
            claim_text = str(unit.get("claim_text") or "").strip()
            time = unit.get("required_time")
            if not claim_text or not isinstance(time, dict) or not isinstance(time.get("required"), bool):
                raise ValueError(f"{claim_id}: invalid claim text/time object")
            required = time["required"]
            resolution = _normalize_time_resolution(time.get("resolution"))
            surface = str(time.get("surface_form") or "").strip().casefold()
            if (
                not required
                or resolution in NON_EXACT_TIME_RESOLUTIONS
                or surface in NON_EXACT_TIME_SURFACES
            ):
                time = {
                    "required": False,
                    "normalized_value": None,
                    "resolution": None,
                    "surface_form": None,
                }
            else:
                time = {**time, "resolution": resolution}
            resolution = time.get("resolution")
            if resolution not in TIME_RESOLUTIONS:
                raise ValueError(f"{claim_id}: invalid time resolution {resolution!r}")
            if time["required"] and resolution is None:
                raise ValueError(f"{claim_id}: required exact time needs a resolution")
            if time["required"] and not str(time.get("surface_form") or "").strip():
                raise ValueError(f"{claim_id}: required time needs surface_form")
            clean_units.append({"claim_id": claim_id, "claim_text": claim_text, "required_time": time})
        normalized.append({"gold_fact_id": fact_id, "original_text": supplied[fact_id], "claim_units": clean_units})
    normalized.sort(key=lambda item: returned_ids.index(item["gold_fact_id"]))
    return {"segment_id": segment_id, "gold_facts": normalized}


def parse_segment_set_judgment(content: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = _json_object(content)
    if payload.get("segment_id") != task["segment_id"]:
        raise ValueError("segment_id mismatch")
    results = payload.get("candidate_set_results")
    if not isinstance(results, list):
        raise ValueError("candidate_set_results must be a list")
    expected_sets = {item["set_id"]: item for item in task["model_input"]["candidate_sets"]}
    returned_sets = [str(item.get("set_id") or "") for item in results if isinstance(item, dict)]
    if len(results) != len(returned_sets) or set(returned_sets) != set(expected_sets) or len(returned_sets) != len(set(returned_sets)):
        raise ValueError("Candidate set IDs are missing, extra, or duplicated")
    required_by_claim: dict[tuple[str, str], bool] = {}
    for fact in task["model_input"]["gold_facts"]:
        for unit in fact["claim_units"]:
            required_by_claim[(fact["gold_fact_id"], unit["claim_id"])] = bool(unit["required_time"]["required"])
    clean_results = []
    for result in results:
        set_id = str(result["set_id"])
        candidate_ids = {str(item["candidate_id"]) for item in expected_sets[set_id]["facts"]}
        assessments = result.get("candidate_assessments")
        if not isinstance(assessments, list):
            raise ValueError(f"{set_id}: candidate_assessments must be a list")
        returned_candidates = [str(item.get("candidate_id") or "") for item in assessments if isinstance(item, dict)]
        if set(returned_candidates) != candidate_ids or len(returned_candidates) != len(candidate_ids):
            raise ValueError(f"{set_id}: Candidate IDs are missing, extra, or duplicated")
        for item in assessments:
            if item.get("semantic_status") not in SEMANTIC_STATUSES:
                raise ValueError(f"{set_id}/{item.get('candidate_id')}: invalid semantic_status")
            _require_string_lists(item, ("supported_content", "unsupported_or_incorrect_content"))
        gold = result.get("gold_claim_assessments")
        if not isinstance(gold, list):
            raise ValueError(f"{set_id}: gold_claim_assessments must be a list")
        returned_claims = [(str(item.get("gold_fact_id") or ""), str(item.get("claim_id") or "")) for item in gold if isinstance(item, dict)]
        if set(returned_claims) != set(required_by_claim) or len(returned_claims) != len(required_by_claim):
            raise ValueError(f"{set_id}: Gold claim IDs are missing, extra, or duplicated")
        for item in gold:
            key = (str(item["gold_fact_id"]), str(item["claim_id"]))
            if item.get("content_status") not in CONTENT_STATUSES:
                raise ValueError(f"{set_id}/{key[1]}: invalid content_status")
            if item.get("time_status") not in TIME_STATUSES:
                raise ValueError(f"{set_id}/{key[1]}: invalid time_status")
            if required_by_claim[key] == (item["time_status"] == "NOT_APPLICABLE"):
                raise ValueError(f"{set_id}/{key[1]}: time_status contradicts required_time")
            covering = item.get("covering_candidate_ids")
            if not isinstance(covering, list) or not set(map(str, covering)).issubset(candidate_ids):
                raise ValueError(f"{set_id}/{key[1]}: invalid covering_candidate_ids")
            if item["time_status"] == "MISSING_REQUIRED_EXACT_TIME":
                statuses = {
                    str(candidate["candidate_id"]): candidate["semantic_status"]
                    for candidate in assessments
                }
                incorrectly_supported = [
                    candidate_id for candidate_id in map(str, covering)
                    if statuses[candidate_id] == "SUPPORTED"
                ]
                if incorrectly_supported:
                    raise ValueError(
                        f"{set_id}/{key[1]}: Candidates missing required exact time "
                        f"cannot be SUPPORTED: {incorrectly_supported}"
                    )
            _require_string_lists(item, ("covered_content", "missing_or_incorrect_content"))
        clean_results.append(result)
    clean_results.sort(key=lambda item: item["set_id"])
    return {"segment_id": task["segment_id"], "candidate_set_results": clean_results}


def _build_set_tasks(segments_path: Path, units_path: Path, candidates_path: Path, seed: int) -> list[dict[str, Any]]:
    segments = _load_segments(segments_path)
    units = {tuple(_identity(row)): row for row in iter_jsonl(units_path)}
    candidates: dict[tuple[str, str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(candidates_path):
        key = tuple(_identity(row))
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        fact_id = str(row.get("fact_id") or row.get("candidate_fact_id") or "").strip()
        text = str(row.get("text") or row.get("fact_text") or "").strip()
        if not model_id or not fact_id or not text:
            raise ValueError("candidate row lacks model_id, fact_id, or text")
        candidates[key][model_id].append({"candidate_id": fact_id, "text": text})
    if set(units) - set(segments):
        raise ValueError(f"Gold units reference missing segments: {list(set(units)-set(segments))[:5]}")
    models = sorted({model for groups in candidates.values() for model in groups})
    if not models:
        raise ValueError("candidate corpus contains no models")
    tasks = []
    for key, unit_row in sorted(units.items()):
        missing = set(models) - set(candidates.get(key, {}))
        if missing:
            raise ValueError(f"segment {key[-1]} lacks candidate sets: {sorted(missing)}")
        ordered_models = list(models)
        random.Random(f"{seed}:{key[-1]}").shuffle(ordered_models)
        model_by_set = {chr(ord("A") + index): model for index, model in enumerate(ordered_models)}
        candidate_sets = [
            {"set_id": set_id, "facts": sorted(candidates[key][model], key=lambda item: item["candidate_id"])}
            for set_id, model in model_by_set.items()
        ]
        segment = segments[key]
        tasks.append({
            "segment_id": key[-1], "dataset_name": key[0], "split": key[1], "sample_id": key[2],
            "model_by_set": model_by_set,
            "model_input": {
                "segment_id": key[-1], "segment_text": segment["text"],
                "gold_facts": unit_row["gold_facts"], "candidate_sets": candidate_sets,
            },
        })
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
        raise ValueError(f"no reference Fact sets found: {path}")
    return sorted(rows, key=_segment_sort_key)


def _validate_gold_units_artifact(units_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "gold_evaluation_unit_manifest_v1":
        raise ValueError("Gold-unit manifest must use gold_evaluation_unit_manifest_v1")
    if manifest.get("run_complete") is not True or manifest.get("status") != "complete":
        raise ValueError("Gold-unit manifest is not complete")
    if manifest.get("output_sha256") != file_sha256(units_path):
        raise ValueError("Gold-unit manifest/output hash mismatch")


def _validate_candidate_artifact(candidates_path: Path, inventory_path: Path) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema_version") not in {"candidate_inventory_v1", "candidate_inventory_v2"}:
        raise ValueError("candidate inventory has an unsupported schema_version")
    if inventory.get("candidate_facts_sha256") != file_sha256(candidates_path):
        raise ValueError("candidate inventory/corpus hash mismatch")


def _normalize_time_resolution(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in NULL_LIKE_TIME_VALUES:
        return None
    return TIME_RESOLUTION_ALIASES.get(normalized, normalized)


def _gold_output_row(
    *, parsed: dict[str, Any], reference_row: dict[str, Any],
    identity: dict[str, Any], model_spec: ModelSpec,
    recovered_from_raw_call: str = "",
) -> dict[str, Any]:
    return {
        **parsed,
        "schema_version": GOLD_SCHEMA_VERSION,
        "dataset_name": reference_row["dataset_name"],
        "dataset": reference_row["dataset_name"],
        "split": reference_row["split"],
        "sample_id": reference_row["sample_id"],
        "reference_set_hash": reference_row.get("reference_set_hash", ""),
        "prompt_version": GOLD_PROMPT_VERSION,
        "prompt_sha256": identity["prompt_sha256"],
        "judge_model": model_spec.effective_model_name,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "recovered_from_raw_call": recovered_from_raw_call,
    }


def _recover_gold_archives(
    *, output_dir: Path, references: list[dict[str, Any]], ledger: SqliteLedger,
    identity: dict[str, Any], model_spec: ModelSpec,
) -> int:
    """Commit newly valid archived responses after a parser-only repair."""
    raw_dir = output_dir / "raw_calls"
    if not raw_dir.is_dir():
        return 0
    by_segment = {str(row["segment_id"]): row for row in references}
    completed = {str(row["segment_id"]) for row in ledger.read_all()}
    recovered = 0
    for path in sorted(raw_dir.glob("*.json"), reverse=True):
        archive = json.loads(path.read_text(encoding="utf-8"))
        segment_id = str(archive.get("segment_id") or "")
        if (
            segment_id in completed
            or segment_id not in by_segment
            or archive.get("status") != "invalid_semantic_response"
            or not str(archive.get("response_content") or "").strip()
        ):
            continue
        try:
            parsed = parse_gold_evaluation_units(
                str(archive["response_content"]), by_segment[segment_id]
            )
        except (ValueError, json.JSONDecodeError):
            continue
        row = _gold_output_row(
            parsed=parsed,
            reference_row=by_segment[segment_id],
            identity=identity,
            model_spec=model_spec,
            recovered_from_raw_call=path.name,
        )
        if ledger.append(row):
            completed.add(segment_id)
            recovered += 1
            print(f"[recover] {segment_id}: committed archived response without an API call")
    return recovered


def _load_segments(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result = {}
    for row in iter_jsonl(path):
        if not {"dataset_name", "split", "sample_id", "segment_id", "text"}.issubset(row):
            continue
        key = tuple(_identity(row))
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"no segments found: {path}")
    return result


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset_name") or row.get("dataset") or ""),
        str(row.get("split") or ""), str(row.get("sample_id") or ""),
        str(row.get("segment_id") or ""),
    )


def _gold_request(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": row["segment_id"],
        "gold_facts": [
            {"gold_fact_id": item["reference_fact_id"], "original_text": item.get("text") or item.get("fact_text")}
            for item in row["reference_facts"]
        ],
    }


def _render(prompt_text: str, payload: dict[str, Any]) -> str:
    return prompt_text.rstrip() + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response root must be an object")
    return value


def _require_string_lists(item: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        value = item.get(field)
        if not isinstance(value, list) or any(not isinstance(element, str) for element in value):
            raise ValueError(f"{field} must be a list of strings")


def _gold_max_tokens(row: dict[str, Any], model: ModelSpec) -> int:
    return min(model.max_output_tokens, max(1024, 320 * len(row["reference_facts"])))


def _set_max_tokens(task: dict[str, Any], model: ModelSpec) -> int:
    candidates = sum(len(item["facts"]) for item in task["model_input"]["candidate_sets"])
    claims = sum(len(item["claim_units"]) for item in task["model_input"]["gold_facts"])
    return min(model.max_output_tokens, max(2048, 260 * (candidates + claims)))


def _plan(*, prompts: list[str], item_count: int, output_reservations: list[int], model_spec: ModelSpec,
          price: PriceSpec, schema_version: str, prompt_version: str, input_hash: str,
          input_hash_name: str, prompt_path: str | Path) -> dict[str, Any]:
    input_tokens = [count_tokens(prompt) for prompt in prompts]
    largest = max(input_tokens, default=0)
    if largest > model_spec.max_input_tokens:
        raise ValueError(f"largest prompt ({largest}) exceeds model input capacity ({model_spec.max_input_tokens})")
    return {
        "schema_version": schema_version, "paid_api_called": False,
        "segment_count": item_count, "logical_api_call_count": item_count,
        "judge_model": model_spec.effective_model_name,
        "prompt_version": prompt_version, "prompt_sha256": file_sha256(prompt_path),
        input_hash_name: input_hash, "estimated_input_tokens": sum(input_tokens),
        "reserved_max_output_tokens": sum(output_reservations),
        "largest_estimated_input_tokens": largest,
        "estimated_upper_bound_input_cost": sum(input_tokens) * price.official_price_in_per_1m / 1_000_000,
        "estimated_upper_bound_output_cost": sum(output_reservations) * price.official_price_out_per_1m / 1_000_000,
        "currency": price.currency, "price_effective_date": price.price_effective_date,
    }


def _manifest(*, identity: dict[str, Any], schema_version: str, total: int, completed: int,
              output_path: Path, output_dir: Path, price: PriceSpec) -> dict[str, Any]:
    usage = _usage_totals(output_dir)
    complete = completed == total
    return {
        "schema_version": schema_version, **identity,
        "status": "complete" if complete else "incomplete", "run_complete": complete,
        "segment_count": total, "completed_segment_count": completed,
        "remaining_segment_count": total - completed,
        "logical_api_call_count": usage["logical_api_call_count"],
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "input_cost": usage["input_tokens"] * price.official_price_in_per_1m / 1_000_000,
        "output_cost": usage["output_tokens"] * price.official_price_out_per_1m / 1_000_000,
        "total_cost": usage["input_tokens"] * price.official_price_in_per_1m / 1_000_000 + usage["output_tokens"] * price.official_price_out_per_1m / 1_000_000,
        "currency": price.currency, "price_effective_date": price.price_effective_date,
        "output": str(output_path.resolve()), "output_sha256": file_sha256(output_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _archive_call(output_dir: Path, segment_id: str, prompt: str, status: str, content: str,
                  response: Any, error: str) -> None:
    raw = output_dir / "raw_calls"
    raw.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = raw / f"{stamp}_{uuid.uuid4().hex[:8]}.json"
    usage = None if response is None else {
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms, "retry_count": response.retry_count,
        "provider_request_id": response.provider_request_id, "finish_reason": response.finish_reason,
    }
    atomic_write_json(path, {
        "schema_version": "segment_set_llm_call_v1", "segment_id": segment_id,
        "status": status, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_content": content, "usage": usage, "error": error,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    })


def _usage_totals(output_dir: Path) -> dict[str, int]:
    totals: Counter[str] = Counter(
        {
            "logical_api_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
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


def _require_resume_identity(path: Path, identity: dict[str, Any]) -> None:
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {key: (existing.get(key), value) for key, value in identity.items() if existing.get(key) != value}
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


def _combined_hash(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        source = Path(path)
        digest.update((_path_digest(source) if source.is_dir() else file_sha256(source)).encode("ascii"))
    return digest.hexdigest()


def _take_stratified(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin samples so a small Pilot is not one conversation prefix."""
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
    return str(row.get("sample_id", "")), int(order or 0), str(row.get("segment_id", ""))

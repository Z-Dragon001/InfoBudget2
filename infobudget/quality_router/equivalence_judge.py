"""Resumable, evidence-grounded batch judging for candidate/Gold Fact pairs."""

from __future__ import annotations

import hashlib
import json
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


PROMPT_VERSION = "fact_equivalence_judge_v1"
SCHEMA_VERSION = "fact_equivalence_judgment_v1"
ALLOWED_REASON_CODES = {
    "EQUIVALENT",
    "CANDIDATE_UNSUPPORTED",
    "REFERENCE_UNSUPPORTED",
    "DIFFERENT_SUBJECT",
    "DIFFERENT_CLAIM",
    "TIME_STATE_MISMATCH",
    "MODALITY_ATTRIBUTION_MISMATCH",
    "POLARITY_MISMATCH",
    "GRANULARITY_MISMATCH",
    "AMBIGUOUS",
}
_TURN_START = re.compile(r"^\[[^\]]+\]\s+\d+\.[^:]+:")
_PAIR_REQUIRED = {
    "pair_id",
    "dataset_name",
    "split",
    "sample_id",
    "segment_id",
    "model_id",
    "candidate_fact_id",
    "candidate_fact_text",
    "candidate_source_turn_ids",
    "reference_fact_id",
    "reference_fact_text",
    "reference_source_turn_ids",
}


def plan_equivalence_judging(
    *,
    segments_path: str | Path,
    pairs_path: str | Path,
    pairs_manifest_path: str | Path | None,
    prompt_path: str | Path,
    model_spec: ModelSpec,
    price: PriceSpec,
    batch_size: int,
) -> dict[str, Any]:
    """Return a read-only call/token/cost plan without contacting a model."""
    pairs = _load_pairs(Path(pairs_path))
    _validate_pairs_manifest(Path(pairs_path), pairs_manifest_path)
    evidence = _load_segment_evidence(Path(segments_path))
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    batches = _build_batches(pairs, batch_size=batch_size)
    input_tokens = 0
    reserved_output_tokens = 0
    largest_input = 0
    for batch in batches:
        prompt = _render_prompt(prompt_text, batch, evidence)
        tokens = count_tokens(prompt)
        input_tokens += tokens
        largest_input = max(largest_input, tokens)
        reserved_output_tokens += _max_new_tokens(len(batch), model_spec)
    if largest_input > model_spec.max_input_tokens:
        raise ValueError(
            f"largest planned prompt ({largest_input}) exceeds judge input capacity "
            f"({model_spec.max_input_tokens})"
        )
    return {
        "schema_version": "fact_equivalence_judge_plan_v1",
        "paid_api_called": False,
        "pair_count": len(pairs),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "judge_model": model_spec.effective_model_name,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": file_sha256(prompt_path),
        "pairs_sha256": file_sha256(pairs_path),
        "segments_sha256": _path_digest(Path(segments_path)),
        "estimated_input_tokens": input_tokens,
        "reserved_max_output_tokens": reserved_output_tokens,
        "largest_estimated_batch_input_tokens": largest_input,
        "estimated_upper_bound_input_cost": (
            input_tokens * price.official_price_in_per_1m / 1_000_000
        ),
        "estimated_upper_bound_output_cost": (
            reserved_output_tokens * price.official_price_out_per_1m / 1_000_000
        ),
        "currency": price.currency,
        "price_effective_date": price.price_effective_date,
        "estimate_note": (
            "Input tokens use the repository's lightweight estimator; output cost "
            "uses the reserved maximum and is therefore an upper bound."
        ),
    }


def run_equivalence_judging(
    *,
    segments_path: str | Path,
    pairs_path: str | Path,
    pairs_manifest_path: str | Path | None,
    prompt_path: str | Path,
    output_dir: str | Path,
    output_path: str | Path,
    model_spec: ModelSpec,
    price: PriceSpec,
    client: ChatCompletionClient,
    batch_size: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Judge remaining pairs, resume from SQLite, and export only when complete."""
    pairs_path = Path(pairs_path)
    prompt_path = Path(prompt_path)
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    pairs = _load_pairs(pairs_path)
    evidence = _load_segment_evidence(Path(segments_path))
    prompt_text = prompt_path.read_text(encoding="utf-8")
    _validate_pairs_manifest(pairs_path, pairs_manifest_path)

    identity = {
        "pairs_sha256": file_sha256(pairs_path),
        "segments_sha256": _path_digest(Path(segments_path)),
        "prompt_sha256": file_sha256(prompt_path),
        "prompt_version": PROMPT_VERSION,
        "judge_model": model_spec.effective_model_name,
        "batch_size": int(batch_size),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    _require_resume_identity(manifest_path, identity)
    ledger = SqliteLedger(
        output_dir / "judgments.sqlite3",
        "judgments",
        key_fields=("pair_id",),
    )
    expected = {str(row["pair_id"]): row for row in pairs}
    completed_rows = ledger.read_all()
    completed = _validate_completed_rows(completed_rows, expected)
    manifest = _make_manifest(
        identity=identity,
        output_dir=output_dir,
        pair_count=len(pairs),
        completed_count=len(completed),
        price=price,
        output_path=None,
    )
    atomic_write_json(manifest_path, manifest)

    remaining = [row for row in pairs if row["pair_id"] not in completed]
    batches = _build_batches(remaining, batch_size=batch_size)
    selected = batches if max_batches is None else batches[:max_batches]
    for batch_number, batch in enumerate(selected, start=1):
        prompt = _render_prompt(prompt_text, batch, evidence)
        max_new_tokens = _max_new_tokens(len(batch), model_spec)
        batch_id = _batch_id(batch)
        try:
            response = client.complete(
                model_spec=model_spec,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                json_mode=True,
            )
        except ModelAPIError as exc:
            _archive_call(
                output_dir,
                batch_id=batch_id,
                batch=batch,
                prompt=prompt,
                status="transport_failed",
                response_content="",
                usage=None,
                error=str(exc),
                attempts=exc.attempts,
            )
            raise

        archive_path = _archive_call(
            output_dir,
            batch_id=batch_id,
            batch=batch,
            prompt=prompt,
            status="response_received",
            response_content=response.content,
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "usage_source": response.usage_source,
                "latency_ms": response.latency_ms,
                "retry_count": response.retry_count,
                "provider_request_id": response.provider_request_id,
                "finish_reason": response.finish_reason,
            },
            error="",
            attempts=response.attempts,
        )
        try:
            decisions = parse_equivalence_decisions(response.content, batch)
        except ValueError as exc:
            failed_payload = json.loads(archive_path.read_text(encoding="utf-8"))
            failed_payload["status"] = "invalid_semantic_response"
            failed_payload["validation_error"] = str(exc)
            atomic_write_json(archive_path, failed_payload)
            raise ValueError(
                f"judge returned an invalid batch {batch_id}; raw response: {archive_path}; {exc}"
            ) from exc

        batch_lookup = {str(row["pair_id"]): row for row in batch}
        judged_at = _utc_now()
        for decision in decisions:
            source = batch_lookup[decision["pair_id"]]
            row = {
                "schema_version": SCHEMA_VERSION,
                "pair_id": decision["pair_id"],
                "dataset": source.get("dataset") or source["dataset_name"],
                "dataset_name": source["dataset_name"],
                "split": source["split"],
                "sample_id": source["sample_id"],
                "segment_id": source["segment_id"],
                "model_id": source["model_id"],
                "candidate_fact_id": source["candidate_fact_id"],
                "reference_fact_id": source["reference_fact_id"],
                **decision,
                "judge_model": model_spec.effective_model_name,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": identity["prompt_sha256"],
                "pairs_sha256": identity["pairs_sha256"],
                "judge_batch_id": batch_id,
                "judged_at": judged_at,
            }
            ledger.append(row)

        completed_rows = ledger.read_all()
        completed = _validate_completed_rows(completed_rows, expected)
        manifest = _make_manifest(
            identity=identity,
            output_dir=output_dir,
            pair_count=len(pairs),
            completed_count=len(completed),
            price=price,
            output_path=None,
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"[{batch_number}/{len(selected)}] {batch_id}: "
            f"{len(batch)} decisions committed; total={len(completed)}/{len(pairs)}",
            flush=True,
        )

    completed_rows = ledger.read_all()
    completed = _validate_completed_rows(completed_rows, expected)
    final_output: Path | None = None
    if len(completed) == len(pairs):
        ordered = [completed[str(pair["pair_id"])] for pair in pairs]
        final_output = write_jsonl(output_path, ordered)
    manifest = _make_manifest(
        identity=identity,
        output_dir=output_dir,
        pair_count=len(pairs),
        completed_count=len(completed),
        price=price,
        output_path=final_output,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_equivalence_decisions(
    content: str, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Strictly validate one model response against the requested pair IDs."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError("response must be an object containing a decisions array")
    expected_ids = [str(row["pair_id"]) for row in batch]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["decisions"]):
        if not isinstance(raw, dict):
            raise ValueError(f"decisions[{index}] must be an object")
        pair_id = str(raw.get("pair_id") or "")
        if not pair_id or pair_id in seen:
            raise ValueError(f"missing or duplicate pair_id at decisions[{index}]")
        seen.add(pair_id)
        flags = {}
        for field in (
            "equivalent",
            "candidate_entailed",
            "reference_entailed",
            "same_claim",
        ):
            if type(raw.get(field)) is not bool:
                raise ValueError(f"{pair_id}: {field} must be a JSON boolean")
            flags[field] = raw[field]
        reason = str(raw.get("reason_code") or "").strip().upper()
        if reason not in ALLOWED_REASON_CODES:
            raise ValueError(f"{pair_id}: unsupported reason_code {reason!r}")
        logically_equivalent = (
            flags["candidate_entailed"]
            and flags["reference_entailed"]
            and flags["same_claim"]
        )
        if flags["equivalent"] != logically_equivalent:
            raise ValueError(
                f"{pair_id}: equivalent/entailment/same_claim/reason_code are inconsistent"
            )
        if (reason == "EQUIVALENT") != flags["equivalent"]:
            raise ValueError(
                f"{pair_id}: EQUIVALENT reason_code must agree with equivalent"
            )
        result.append({"pair_id": pair_id, **flags, "reason_code": reason})
    if set(expected_ids) != seen or len(result) != len(expected_ids):
        missing = sorted(set(expected_ids) - seen)
        extra = sorted(seen - set(expected_ids))
        raise ValueError(f"decision ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    by_id = {row["pair_id"]: row for row in result}
    return [by_id[pair_id] for pair_id in expected_ids]


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        missing = sorted(_PAIR_REQUIRED - set(row))
        if missing:
            raise ValueError(f"pair row is missing fields: {missing}")
        pair_id = str(row["pair_id"])
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"no equivalence pairs found: {path}")
    return rows


def _load_segment_evidence(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[int, str]]:
    result: dict[tuple[str, str, str, str], dict[int, str]] = {}
    for row in iter_jsonl(path):
        if not {"dataset_name", "split", "sample_id", "segment_id", "turn_ids", "text"}.issubset(row):
            continue
        key = (
            str(row["dataset_name"]),
            str(row["split"]),
            str(row["sample_id"]),
            str(row["segment_id"]),
        )
        if key in result:
            raise ValueError(f"duplicate segment evidence: {key}")
        turn_ids = [int(value) for value in row["turn_ids"]]
        chunks = _split_turn_text(str(row["text"]))
        if len(turn_ids) != len(chunks):
            raise ValueError(
                f"cannot align source turns for {key}: ids={len(turn_ids)}, text={len(chunks)}"
            )
        result[key] = {
            turn_id: f"<SOURCE_TURN_ID={turn_id}> {text}"
            for turn_id, text in zip(turn_ids, chunks)
        }
    if not result:
        raise ValueError(f"no segment evidence found: {path}")
    return result


def _split_turn_text(text: str) -> list[str]:
    chunks: list[str] = []
    for line in text.splitlines():
        if _TURN_START.match(line):
            chunks.append(line)
        elif chunks:
            chunks[-1] += "\n" + line
        elif line.strip():
            raise ValueError("segment text begins with an unrecognized continuation line")
    return chunks


def _build_batches(
    pairs: list[dict[str, Any]], *, batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]


def _render_prompt(
    prompt_text: str,
    batch: list[dict[str, Any]],
    evidence: dict[tuple[str, str, str, str], dict[int, str]],
) -> str:
    requested_by_segment: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for row in batch:
        key = (
            str(row["dataset_name"]),
            str(row["split"]),
            str(row["sample_id"]),
            str(row["segment_id"]),
        )
        for field in ("candidate_source_turn_ids", "reference_source_turn_ids"):
            requested_by_segment[key].update(int(value) for value in row[field])
    evidence_payload = []
    for key in sorted(requested_by_segment):
        source_map = evidence.get(key)
        if source_map is None:
            raise ValueError(f"missing evidence segment: {key}")
        requested_turns = sorted(requested_by_segment[key])
        missing = [turn_id for turn_id in requested_turns if turn_id not in source_map]
        if missing:
            raise ValueError(f"pair source IDs are outside segment {key}: {missing}")
        evidence_payload.append(
            {
                "dataset_name": key[0],
                "split": key[1],
                "sample_id": key[2],
                "segment_id": key[3],
                "turns": [source_map[turn_id] for turn_id in requested_turns],
            }
        )
    payload = {
        "source_turn_evidence_by_segment": evidence_payload,
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "dataset_name": row["dataset_name"],
                "split": row["split"],
                "sample_id": row["sample_id"],
                "segment_id": row["segment_id"],
                "candidate_fact": row["candidate_fact_text"],
                "candidate_source_turn_ids": row["candidate_source_turn_ids"],
                "reference_fact": row["reference_fact_text"],
                "reference_source_turn_ids": row["reference_source_turn_ids"],
            }
            for row in batch
        ],
    }
    return prompt_text.rstrip() + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def _max_new_tokens(batch_length: int, model_spec: ModelSpec) -> int:
    return min(model_spec.max_output_tokens, max(512, batch_length * 96 + 256))


def _batch_id(batch: list[dict[str, Any]]) -> str:
    payload = [str(row["pair_id"]) for row in batch]
    digest = hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
    return f"eqj_{digest[:20]}"


def _archive_call(
    output_dir: Path,
    *,
    batch_id: str,
    batch: list[dict[str, Any]],
    prompt: str,
    status: str,
    response_content: str,
    usage: dict[str, Any] | None,
    error: str,
    attempts: Any,
) -> Path:
    archive_dir = output_dir / "raw_calls"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = archive_dir / f"{stamp}_{batch_id}_{uuid.uuid4().hex[:8]}.json"
    atomic_write_json(
        path,
        {
            "schema_version": "fact_equivalence_judge_raw_call_v1",
            "status": status,
            "batch_id": batch_id,
            "pair_ids": [row["pair_id"] for row in batch],
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_content": response_content,
            "usage": usage,
            "error": error,
            "transport_attempts": attempts or [],
            "archived_at": _utc_now(),
        },
    )
    return path


def _validate_pairs_manifest(
    pairs_path: Path, manifest_path: str | Path | None
) -> None:
    if manifest_path is None:
        return
    source = Path(manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = str(payload.get("output_sha256") or "")
    actual = file_sha256(pairs_path)
    if not expected or expected != actual:
        raise ValueError(
            f"pair manifest hash mismatch: manifest={expected!r}, actual={actual}"
        )


def _require_resume_identity(path: Path, identity: dict[str, Any]) -> None:
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    mismatched = {
        key: (previous.get(key), value)
        for key, value in identity.items()
        if previous.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "output directory belongs to a different judge run: "
            + json.dumps(mismatched, ensure_ascii=False, sort_keys=True)
        )


def _validate_completed_rows(
    rows: Iterable[dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id not in expected:
            raise ValueError(f"ledger contains an unknown pair_id: {pair_id}")
        if pair_id in result:
            raise ValueError(f"ledger contains duplicate pair_id: {pair_id}")
        result[pair_id] = row
    return result


def _make_manifest(
    *,
    identity: dict[str, Any],
    output_dir: Path,
    pair_count: int,
    completed_count: int,
    price: PriceSpec,
    output_path: Path | None,
) -> dict[str, Any]:
    raw_calls = []
    for path in sorted((output_dir / "raw_calls").glob("*.json")):
        raw_calls.append(json.loads(path.read_text(encoding="utf-8")))
    successful = [row for row in raw_calls if row.get("status") in {"response_received", "invalid_semantic_response"} and isinstance(row.get("usage"), dict)]
    input_tokens = sum(int(row["usage"].get("input_tokens") or 0) for row in successful)
    output_tokens = sum(int(row["usage"].get("output_tokens") or 0) for row in successful)
    reason_counts: Counter[str] = Counter()
    ledger_path = output_dir / "judgments.sqlite3"
    if ledger_path.is_file():
        rows = SqliteLedger(ledger_path, "judgments", key_fields=("pair_id",)).read_all()
        reason_counts.update(str(row.get("reason_code") or "") for row in rows)
    complete = completed_count == pair_count
    return {
        "schema_version": "fact_equivalence_judge_manifest_v1",
        **identity,
        "status": "complete" if complete else "incomplete",
        "run_complete": complete,
        "pair_count": pair_count,
        "completed_decision_count": completed_count,
        "remaining_decision_count": pair_count - completed_count,
        "logical_api_call_count": len(raw_calls),
        "successful_response_count": len(successful),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_tokens * price.official_price_in_per_1m / 1_000_000,
        "output_cost": output_tokens * price.official_price_out_per_1m / 1_000_000,
        "total_cost": (
            input_tokens * price.official_price_in_per_1m
            + output_tokens * price.official_price_out_per_1m
        ) / 1_000_000,
        "currency": price.currency,
        "price_effective_date": price.price_effective_date,
        "reason_counts": dict(sorted(reason_counts.items())),
        "output": str(output_path.resolve()) if output_path else None,
        "output_sha256": file_sha256(output_path) if output_path else None,
        "updated_at": _utc_now(),
    }


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found: {path}")
    for item in paths:
        relative = item.name if path.is_file() else str(item.relative_to(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

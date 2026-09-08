"""Coverage audit for the frozen candidate/Gold Fact pair universe."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json


SegmentKey = tuple[str, str, str, str]
CandidateKey = tuple[str, str, str, str, str, str]
ReferenceKey = tuple[str, str, str, str, str]
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", re.UNICODE)


def audit_fact_pair_coverage(
    *,
    candidates_path: str | Path,
    references_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
    risky_pairs_output_path: str | Path,
    lexical_threshold: float = 0.35,
    sequence_threshold: float = 0.72,
    adjacent_turn_distance: int = 1,
) -> dict[str, Any]:
    """Measure unique-Fact coverage and export potentially missed excluded pairs."""
    if not 0.0 <= lexical_threshold <= 1.0:
        raise ValueError("lexical_threshold must be between 0 and 1")
    if not 0.0 <= sequence_threshold <= 1.0:
        raise ValueError("sequence_threshold must be between 0 and 1")
    if adjacent_turn_distance < 0:
        raise ValueError("adjacent_turn_distance cannot be negative")

    candidates, candidates_by_segment_model = _load_candidates(Path(candidates_path))
    references, references_by_segment = _load_references(Path(references_path))
    models = sorted({key[4] for key in candidates})
    if not models:
        raise ValueError("candidate corpus contains no model IDs")

    covered_candidates: set[CandidateKey] = set()
    covered_references_by_model: dict[str, set[ReferenceKey]] = defaultdict(set)
    pair_counts_by_candidate: Counter[CandidateKey] = Counter()
    pair_counts_by_reference_model: Counter[tuple[str, ReferenceKey]] = Counter()
    pair_counts_by_model: Counter[str] = Counter()
    seen_pair_ids: set[str] = set()

    for row in iter_jsonl(pairs_path):
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError(f"missing or duplicate pair_id: {pair_id!r}")
        seen_pair_ids.add(pair_id)
        segment = _segment_key(row)
        model_id = str(row.get("model_id") or "").strip()
        candidate_key = (*segment, model_id, str(row.get("candidate_fact_id") or ""))
        reference_key = (*segment, str(row.get("reference_fact_id") or ""))
        if candidate_key not in candidates:
            raise ValueError(f"pair references unknown candidate: {candidate_key}")
        if reference_key not in references:
            raise ValueError(f"pair references unknown Gold Fact: {reference_key}")
        candidate_sources = candidates[candidate_key]["source_turn_ids"]
        reference_sources = references[reference_key]["source_turn_ids"]
        if not set(candidate_sources) & set(reference_sources):
            raise ValueError(f"eligible pair has no source overlap: {pair_id}")
        covered_candidates.add(candidate_key)
        covered_references_by_model[model_id].add(reference_key)
        pair_counts_by_candidate[candidate_key] += 1
        pair_counts_by_reference_model[(model_id, reference_key)] += 1
        pair_counts_by_model[model_id] += 1

    raw_pair_count = 0
    for (segment, model_id), values in candidates_by_segment_model.items():
        raw_pair_count += len(values) * len(references_by_segment.get(segment, ()))
    eligible_pair_count = len(seen_pair_ids)
    excluded_pair_count = raw_pair_count - eligible_pair_count
    if excluded_pair_count < 0:
        raise ValueError("eligible pair count exceeds the same-segment Cartesian universe")

    candidate_coverage_by_model: dict[str, dict[str, Any]] = {}
    reference_coverage_by_model: dict[str, dict[str, Any]] = {}
    for model_id in models:
        model_candidates = [key for key in candidates if key[4] == model_id]
        covered_model_candidates = [key for key in model_candidates if key in covered_candidates]
        candidate_coverage_by_model[model_id] = _coverage(
            len(model_candidates), len(covered_model_candidates)
        )
        reference_coverage_by_model[model_id] = _coverage(
            len(references), len(covered_references_by_model.get(model_id, set()))
        )

    risky_rows: list[dict[str, Any]] = []
    risk_reasons: Counter[str] = Counter()
    risky_by_model: Counter[str] = Counter()
    for (segment, model_id), candidate_keys in candidates_by_segment_model.items():
        for candidate_key in candidate_keys:
            candidate = candidates[candidate_key]
            for reference_key in references_by_segment.get(segment, ()):
                reference = references[reference_key]
                if set(candidate["source_turn_ids"]) & set(reference["source_turn_ids"]):
                    continue
                distance = _minimum_source_distance(
                    candidate["source_turn_ids"], reference["source_turn_ids"]
                )
                jaccard = _token_jaccard(candidate["text"], reference["text"])
                sequence_ratio = SequenceMatcher(
                    None, _normalized_text(candidate["text"]), _normalized_text(reference["text"])
                ).ratio()
                exact_normalized = (
                    _normalized_text(candidate["text"])
                    == _normalized_text(reference["text"])
                )
                reasons: list[str] = []
                if exact_normalized:
                    reasons.append("normalized_exact_text")
                if distance is not None and distance <= adjacent_turn_distance:
                    reasons.append("adjacent_source_turns")
                if jaccard >= lexical_threshold:
                    reasons.append("high_token_jaccard")
                if sequence_ratio >= sequence_threshold:
                    reasons.append("high_sequence_similarity")
                if not reasons:
                    continue
                for reason in reasons:
                    risk_reasons[reason] += 1
                risky_by_model[model_id] += 1
                risky_rows.append(
                    {
                        "schema_version": "excluded_fact_pair_review_v1",
                        "dataset_name": segment[0],
                        "split": segment[1],
                        "sample_id": segment[2],
                        "segment_id": segment[3],
                        "model_id": model_id,
                        "candidate_fact_id": candidate_key[5],
                        "candidate_fact_text": candidate["text"],
                        "candidate_source_turn_ids": candidate["source_turn_ids"],
                        "reference_fact_id": reference_key[4],
                        "reference_fact_text": reference["text"],
                        "reference_source_turn_ids": reference["source_turn_ids"],
                        "minimum_source_turn_distance": distance,
                        "token_jaccard": round(jaccard, 6),
                        "sequence_similarity": round(sequence_ratio, 6),
                        "risk_reasons": reasons,
                    }
                )

    risky_rows.sort(
        key=lambda row: (
            -int("normalized_exact_text" in row["risk_reasons"]),
            -max(float(row["token_jaccard"]), float(row["sequence_similarity"])),
            row["minimum_source_turn_distance"]
            if row["minimum_source_turn_distance"] is not None
            else 10**9,
            row["model_id"],
            row["sample_id"],
            row["segment_id"],
            row["candidate_fact_id"],
            row["reference_fact_id"],
        )
    )
    risky_output = write_jsonl(risky_pairs_output_path, risky_rows)

    audit = {
        "schema_version": "fact_pair_coverage_audit_v1",
        "candidates_sha256": file_sha256(candidates_path),
        "references_sha256": file_sha256(references_path),
        "pairs_sha256": file_sha256(pairs_path),
        "candidate_fact_count": len(candidates),
        "reference_fact_count": len(references),
        "model_count": len(models),
        "models": models,
        "raw_same_segment_pair_count": raw_pair_count,
        "eligible_source_overlap_pair_count": eligible_pair_count,
        "excluded_no_source_overlap_pair_count": excluded_pair_count,
        "unique_candidate_with_pair_count": len(covered_candidates),
        "unique_candidate_without_pair_count": len(candidates) - len(covered_candidates),
        "candidate_coverage": _coverage(len(candidates), len(covered_candidates)),
        "candidate_coverage_by_model": candidate_coverage_by_model,
        "reference_coverage_by_model": reference_coverage_by_model,
        "eligible_pairs_by_model": dict(sorted(pair_counts_by_model.items())),
        "eligible_pairs_per_candidate": _distribution(
            [pair_counts_by_candidate.get(key, 0) for key in candidates]
        ),
        "eligible_pairs_per_reference_by_model": {
            model_id: _distribution(
                [
                    pair_counts_by_reference_model.get((model_id, reference_key), 0)
                    for reference_key in references
                ]
            )
            for model_id in models
        },
        "risky_excluded_pair_count": len(risky_rows),
        "risky_excluded_pairs_by_model": dict(sorted(risky_by_model.items())),
        "risk_reason_counts": dict(sorted(risk_reasons.items())),
        "risk_heuristics": {
            "lexical_threshold": lexical_threshold,
            "sequence_threshold": sequence_threshold,
            "adjacent_turn_distance": adjacent_turn_distance,
            "note": "Risk flags are review candidates, not semantic-equivalence labels.",
        },
        "risky_pairs_output": str(risky_output.resolve()),
        "risky_pairs_output_sha256": file_sha256(risky_output),
        "judge_should_run_now": False,
        "next_gate": (
            "Review unique zero-pair coverage and a stratified sample of the risky "
            "excluded queue before freezing the final pair policy."
        ),
    }
    atomic_write_json(output_path, audit)
    return audit


def _load_candidates(
    path: Path,
) -> tuple[
    dict[CandidateKey, dict[str, Any]],
    dict[tuple[SegmentKey, str], list[CandidateKey]],
]:
    result: dict[CandidateKey, dict[str, Any]] = {}
    grouped: dict[tuple[SegmentKey, str], list[CandidateKey]] = defaultdict(list)
    for row in iter_jsonl(path):
        segment = _segment_key(row)
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        fact_id = str(row.get("candidate_fact_id") or row.get("fact_id") or "").strip()
        if not model_id or not fact_id:
            raise ValueError("candidate row is missing model_id or fact_id")
        key = (*segment, model_id, fact_id)
        if key in result:
            raise ValueError(f"duplicate candidate Fact: {key}")
        result[key] = {
            "text": _fact_text(row),
            "source_turn_ids": _source_ids(row),
        }
        grouped[(segment, model_id)].append(key)
    if not result:
        raise ValueError(f"no candidate Facts found: {path}")
    return result, dict(grouped)


def _load_references(
    path: Path,
) -> tuple[
    dict[ReferenceKey, dict[str, Any]],
    dict[SegmentKey, list[ReferenceKey]],
]:
    result: dict[ReferenceKey, dict[str, Any]] = {}
    grouped: dict[SegmentKey, list[ReferenceKey]] = defaultdict(list)
    for row in iter_jsonl(path):
        segment = _segment_key(row)
        for fact in row.get("reference_facts", ()):
            fact_id = str(fact.get("reference_fact_id") or fact.get("fact_id") or "").strip()
            if not fact_id:
                raise ValueError(f"reference Fact is missing an ID: {segment}")
            key = (*segment, fact_id)
            if key in result:
                raise ValueError(f"duplicate reference Fact: {key}")
            result[key] = {
                "text": _fact_text(fact),
                "source_turn_ids": _source_ids(fact),
            }
            grouped[segment].append(key)
    if not result:
        raise ValueError(f"no reference Facts found: {path}")
    return result, dict(grouped)


def _segment_key(row: dict[str, Any]) -> SegmentKey:
    values = (
        str(row.get("dataset_name") or row.get("dataset") or "").strip(),
        str(row.get("split") or "").strip(),
        str(row.get("sample_id") or "").strip(),
        str(row.get("segment_id") or "").strip(),
    )
    if not all(values):
        raise ValueError(f"row is missing segment identity: {values}")
    return values


def _fact_text(row: dict[str, Any]) -> str:
    value = str(
        row.get("candidate_fact_text")
        or row.get("reference_fact_text")
        or row.get("fact_text")
        or row.get("text")
        or ""
    ).strip()
    if not value:
        raise ValueError("Fact text cannot be empty")
    return value


def _source_ids(row: dict[str, Any]) -> list[int]:
    values = row.get("source_turn_ids")
    if values is None:
        values = row.get("candidate_source_turn_ids")
    if values is None:
        values = row.get("reference_source_turn_ids")
    result = sorted({int(value) for value in values or ()})
    if not result:
        raise ValueError("Fact source_turn_ids cannot be empty")
    return result


def _coverage(total: int, covered: int) -> dict[str, Any]:
    return {
        "total": total,
        "with_eligible_pair": covered,
        "without_eligible_pair": total - covered,
        "coverage_rate": covered / total if total else 1.0,
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "median": 0, "p95": 0, "max": 0, "mean": 0.0}
    return {
        "min": ordered[0],
        "median": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _percentile(ordered: list[int], quantile: float) -> int:
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _minimum_source_distance(left: list[int], right: list[int]) -> int | None:
    if not left or not right:
        return None
    return min(abs(a - b) for a in left for b in right)


def _normalized_text(text: str) -> str:
    return " ".join(token.casefold() for token in _TOKEN.findall(text))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_normalized_text(left).split())
    right_tokens = set(_normalized_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


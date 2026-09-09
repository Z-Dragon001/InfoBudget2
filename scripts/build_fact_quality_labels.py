"""Build partial-credit quality labels from completed Segment-set judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import (
    file_sha256, iter_jsonl, load_capability_profiles, write_jsonl,
)
from infobudget.quality_router.segment_set_evaluation import gold_requires_exact_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-decisions", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    args = parser.parse_args()

    _validate_manifest(
        args.judge_manifest, args.judge_decisions,
        args.references, args.candidates,
    )
    profiles = load_capability_profiles(args.capabilities)
    references = {
        str(row["segment_id"]): row for row in iter_jsonl(args.references)
        if isinstance(row.get("reference_facts"), list)
    }
    run_ids = _candidate_run_ids(args.candidates)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for judgment in iter_jsonl(args.judge_decisions):
        segment_id = str(judgment["segment_id"])
        reference_row = references.get(segment_id)
        if reference_row is None:
            raise ValueError(f"Judge references unknown Gold segment: {segment_id}")
        gold_time = {
            str(fact["reference_fact_id"]): gold_requires_exact_time(
                str(fact.get("text") or fact.get("fact_text"))
            )
            for fact in reference_row["reference_facts"]
        }
        for result in judgment["candidate_set_results"]:
            model_id = str(result["model_id"])
            key = (segment_id, model_id)
            if key in seen:
                raise ValueError(f"duplicate segment/model judgment: {key}")
            seen.add(key)
            if model_id not in profiles:
                raise ValueError(f"capability profile is missing model: {model_id}")
            label, detail = _score_result(
                judgment=judgment, result=result, gold_time=gold_time,
                profile_id=profiles[model_id].profile_id,
                candidate_extraction_run_id=run_ids.get(key, "unknown"),
                reference_set_hash=str(
                    reference_row.get("reference_set_hash") or "unknown"
                ),
            )
            rows.append(label)
            details.append(detail)
    rows.sort(key=_row_key)
    details.sort(key=_row_key)
    write_jsonl(args.output, rows)
    if args.details_output:
        write_jsonl(args.details_output, details)
    print(json.dumps({
        "schema_version": "segment_set_quality_label_build_v2",
        "label_count": len(rows),
        "segment_count": len({row["segment_id"] for row in rows}),
        "model_count": len({row["model_id"] for row in rows}),
        "primary_label": "set_quality_f2",
        "partial_credit": 0.5,
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
    }, ensure_ascii=False, indent=2))


def _score_result(
    *, judgment: dict[str, Any], result: dict[str, Any],
    gold_time: dict[str, bool], profile_id: str,
    candidate_extraction_run_id: str, reference_set_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_values = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "UNSUPPORTED": 0.0,
        "CONTRADICTED": 0.0,
    }
    candidates = result["candidate_assessments"]
    supported = sum(item["semantic_status"] == "SUPPORTED" for item in candidates)
    partial_candidates = sum(
        item["semantic_status"] == "PARTIALLY_SUPPORTED" for item in candidates
    )
    candidate_total = len(candidates)
    strict_precision = supported / candidate_total if candidate_total else 1.0
    soft_precision = (
        sum(candidate_values[item["semantic_status"]] for item in candidates)
        / candidate_total if candidate_total else 1.0
    )

    assessments = {
        str(item["gold_fact_id"]): item
        for item in result["gold_fact_assessments"]
    }
    if set(assessments) != set(gold_time):
        raise ValueError(
            f"{judgment['segment_id']}/{result['model_id']}: Gold Fact mismatch"
        )
    strict_full: set[str] = set()
    weighted_sum = 0.0
    temporal_pass = 0
    temporal_total = sum(gold_time.values())
    for gold_id, item in assessments.items():
        time_ok = item["time_status"] in {"PASS", "NOT_APPLICABLE"}
        if gold_time[gold_id] and item["time_status"] == "PASS":
            temporal_pass += 1
        if not time_ok:
            continue
        if item["coverage_status"] == "FULL":
            strict_full.add(gold_id)
            weighted_sum += 1.0
        elif item["coverage_status"] == "PARTIAL":
            weighted_sum += 0.5
    gold_total = len(gold_time)
    strict_recall = len(strict_full) / gold_total if gold_total else 1.0
    weighted_recall = weighted_sum / gold_total if gold_total else 1.0
    strict_f1 = _fbeta(strict_precision, strict_recall, beta=1.0)
    quality_f2 = _fbeta(soft_precision, weighted_recall, beta=2.0)
    temporal_recall = temporal_pass / temporal_total if temporal_total else None
    identity = {
        "dataset": judgment.get("dataset_name") or judgment.get("dataset"),
        "split": judgment["split"], "sample_id": judgment["sample_id"],
        "segment_id": judgment["segment_id"], "model_id": result["model_id"],
    }
    label = {
        "schema_version": "segment_set_quality_label_v2", **identity,
        "profile_id": profile_id,
        "tp": supported, "fp": candidate_total - supported,
        "fn": gold_total - len(strict_full),
        "precision": strict_precision, "recall": strict_recall,
        "silver_strict_fact_f1": strict_f1,
        "silver_gold_coverage": weighted_recall,
        "set_quality_f2": quality_f2,
        "strict_candidate_precision": strict_precision,
        "partial_credit_candidate_precision": soft_precision,
        "strict_gold_fact_recall": strict_recall,
        "partial_credit_gold_coverage": weighted_recall,
        "temporal_recall": temporal_recall,
        "candidate_fact_count": candidate_total,
        "supported_candidate_count": supported,
        "partially_supported_candidate_count": partial_candidates,
        "fully_covered_gold_fact_count": len(strict_full),
        "gold_fact_count": gold_total,
        "temporal_gold_fact_count": temporal_total,
        "primary_label_name": "set_quality_f2",
        "reference_set_hash": reference_set_hash,
        "candidate_extraction_run_id": candidate_extraction_run_id,
        "label_version": "segment_set_gold_fact_partial_credit_v2",
        "covered_gold_count": len(strict_full),
        "uncovered_gold_count": gold_total - len(strict_full),
        "covering_candidate_count": len({
            candidate_id for gold_id, item in assessments.items()
            if gold_id in strict_full
            for candidate_id in item["covering_candidate_ids"]
        }),
    }
    detail = {
        **identity, "set_id": result["set_id"],
        "candidate_assessments": candidates,
        "gold_fact_assessments": result["gold_fact_assessments"],
    }
    return label, detail


def _fbeta(precision: float, recall: float, *, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return (
        (1 + beta * beta) * precision * recall / denominator
        if denominator else 0.0
    )


def _candidate_run_ids(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in iter_jsonl(path):
        key = (
            str(row["segment_id"]),
            str(row.get("model_id") or row.get("extractor_model")),
        )
        run_id = str(
            row.get("candidate_extraction_run_id")
            or row.get("extraction_run_id") or "unknown"
        )
        previous = result.setdefault(key, run_id)
        if previous != run_id:
            raise ValueError(
                f"candidate segment/model mixes extraction runs: {key}"
            )
    return result


def _validate_manifest(
    manifest_path: Path, decisions_path: Path,
    references_path: Path, candidates_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "segment_fact_set_judge_manifest_v2":
        raise ValueError(
            "Judge manifest must use segment_fact_set_judge_manifest_v2"
        )
    if manifest.get("run_complete") is not True or manifest.get("status") != "complete":
        raise ValueError("set-Judge manifest is not complete")
    checks = {
        "output_sha256": file_sha256(decisions_path),
        "references_sha256": file_sha256(references_path),
        "candidates_sha256": file_sha256(candidates_path),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in checks.items() if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"set-Judge artifact hash mismatch: {mismatches}")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["sample_id"], row["segment_id"], row["model_id"]


if __name__ == "__main__":
    main()

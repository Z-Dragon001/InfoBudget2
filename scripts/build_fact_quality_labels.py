"""Build segment-set quality labels from completed set-Judge decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import file_sha256, iter_jsonl, load_capability_profiles, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-decisions", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--gold-units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    args = parser.parse_args()

    _validate_manifest(args.judge_manifest, args.judge_decisions, args.gold_units, args.candidates)
    profiles = load_capability_profiles(args.capabilities)
    units = {str(row["segment_id"]): row for row in iter_jsonl(args.gold_units)}
    run_ids = _candidate_run_ids(args.candidates)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for judgment in iter_jsonl(args.judge_decisions):
        segment_id = str(judgment["segment_id"])
        if segment_id not in units:
            raise ValueError(f"Judge references unknown Gold-unit segment: {segment_id}")
        gold_units = units[segment_id]
        claims_by_fact = {
            str(fact["gold_fact_id"]): [str(unit["claim_id"]) for unit in fact["claim_units"]]
            for fact in gold_units["gold_facts"]
        }
        required_time = {
            str(unit["claim_id"]): bool(unit["required_time"]["required"])
            for fact in gold_units["gold_facts"] for unit in fact["claim_units"]
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
                judgment=judgment, result=result, claims_by_fact=claims_by_fact,
                required_time=required_time, profile_id=profiles[model_id].profile_id,
                candidate_extraction_run_id=run_ids.get(key, "unknown"),
                reference_set_hash=str(gold_units.get("reference_set_hash") or "unknown"),
            )
            rows.append(label)
            details.append(detail)
    rows.sort(key=lambda row: (row["sample_id"], row["segment_id"], row["model_id"]))
    details.sort(key=lambda row: (row["sample_id"], row["segment_id"], row["model_id"]))
    write_jsonl(args.output, rows)
    if args.details_output:
        write_jsonl(args.details_output, details)
    print(json.dumps({
        "schema_version": "segment_set_quality_label_build_v1",
        "label_count": len(rows), "segment_count": len({row["segment_id"] for row in rows}),
        "model_count": len({row["model_id"] for row in rows}),
        "primary_label": "set_quality_f2", "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
    }, ensure_ascii=False, indent=2))


def _score_result(*, judgment: dict[str, Any], result: dict[str, Any],
                  claims_by_fact: dict[str, list[str]], required_time: dict[str, bool],
                  profile_id: str, candidate_extraction_run_id: str,
                  reference_set_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_assessments = result["candidate_assessments"]
    supported = sum(item["semantic_status"] == "SUPPORTED" for item in candidate_assessments)
    partial_candidates = sum(item["semantic_status"] == "PARTIALLY_SUPPORTED" for item in candidate_assessments)
    candidate_total = len(candidate_assessments)
    precision = supported / candidate_total if candidate_total else 1.0
    soft_precision = (supported + 0.5 * partial_candidates) / candidate_total if candidate_total else 1.0

    claim_rows = {str(item["claim_id"]): item for item in result["gold_claim_assessments"]}
    expected_claims = set(required_time)
    if set(claim_rows) != expected_claims:
        raise ValueError(f"{judgment['segment_id']}/{result['model_id']}: Gold claim set mismatch")
    strict_covered: set[str] = set()
    soft_coverage = 0.0
    temporal_pass = 0
    temporal_total = sum(required_time.values())
    for claim_id, item in claim_rows.items():
        time_ok = item["time_status"] in {"PASS", "NOT_APPLICABLE"}
        if item["content_status"] == "COVERED" and time_ok:
            strict_covered.add(claim_id)
            soft_coverage += 1.0
        elif item["content_status"] == "PARTIAL" and time_ok:
            soft_coverage += 0.5
        if required_time[claim_id] and item["time_status"] == "PASS":
            temporal_pass += 1
    claim_total = len(expected_claims)
    recall = len(strict_covered) / claim_total if claim_total else 1.0
    soft_recall = soft_coverage / claim_total if claim_total else 1.0
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)
    fully_covered_facts = sum(
        bool(claim_ids) and set(claim_ids).issubset(strict_covered)
        for claim_ids in claims_by_fact.values()
    )
    gold_fact_total = len(claims_by_fact)
    gold_fact_recall = fully_covered_facts / gold_fact_total if gold_fact_total else 1.0
    temporal_recall = temporal_pass / temporal_total if temporal_total else None
    identity = {
        "dataset": judgment.get("dataset_name") or judgment.get("dataset"),
        "split": judgment["split"], "sample_id": judgment["sample_id"],
        "segment_id": judgment["segment_id"], "model_id": result["model_id"],
    }
    label = {
        "schema_version": "segment_set_quality_label_v1", **identity,
        "profile_id": profile_id,
        "tp": supported, "fp": candidate_total - supported,
        "fn": claim_total - len(strict_covered),
        "precision": precision, "recall": recall,
        "silver_strict_fact_f1": f1,
        "silver_gold_coverage": recall,
        "set_quality_f2": f2,
        "strict_candidate_precision": precision,
        "soft_candidate_precision": soft_precision,
        "strict_claim_recall": recall,
        "soft_claim_recall": soft_recall,
        "strict_gold_fact_recall": gold_fact_recall,
        "temporal_recall": temporal_recall,
        "candidate_fact_count": candidate_total,
        "supported_candidate_count": supported,
        "strictly_covered_claim_count": len(strict_covered),
        "gold_claim_count": claim_total,
        "fully_covered_gold_fact_count": fully_covered_facts,
        "gold_fact_count": gold_fact_total,
        "temporal_gold_claim_count": temporal_total,
        "primary_label_name": "set_quality_f2",
        "reference_set_hash": reference_set_hash,
        "candidate_extraction_run_id": candidate_extraction_run_id,
        "label_version": "segment_set_semantic_temporal_v1",
        # Compatibility fields for existing readers; semantics are documented above.
        "covered_gold_count": len(strict_covered),
        "uncovered_gold_count": claim_total - len(strict_covered),
        "covering_candidate_count": len({candidate_id for item in claim_rows.values() if item["claim_id"] in strict_covered for candidate_id in item["covering_candidate_ids"]}),
    }
    detail = {**identity, "set_id": result["set_id"], "candidate_assessments": candidate_assessments,
              "gold_claim_assessments": result["gold_claim_assessments"]}
    return label, detail


def _fbeta(precision: float, recall: float, *, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def _candidate_run_ids(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in iter_jsonl(path):
        key = (str(row["segment_id"]), str(row.get("model_id") or row.get("extractor_model")))
        run_id = str(row.get("candidate_extraction_run_id") or row.get("extraction_run_id") or "unknown")
        previous = result.setdefault(key, run_id)
        if previous != run_id:
            raise ValueError(f"candidate segment/model mixes extraction runs: {key}")
    return result


def _validate_manifest(manifest_path: Path, decisions_path: Path, units_path: Path, candidates_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "segment_fact_set_judge_manifest_v1":
        raise ValueError("Judge manifest must use segment_fact_set_judge_manifest_v1")
    if manifest.get("run_complete") is not True or manifest.get("status") != "complete":
        raise ValueError("set-Judge manifest is not complete")
    checks = {
        "output_sha256": file_sha256(decisions_path),
        "gold_units_sha256": file_sha256(units_path),
        "candidates_sha256": file_sha256(candidates_path),
    }
    mismatches = {key: (manifest.get(key), value) for key, value in checks.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"set-Judge artifact hash mismatch: {mismatches}")


if __name__ == "__main__":
    main()

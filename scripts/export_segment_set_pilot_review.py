"""Export readable, source-ID-free rows for Segment-set Pilot review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    segments = {str(row["segment_id"]): str(row["text"]) for row in iter_jsonl(args.segments) if "segment_id" in row and "text" in row}
    references = {
        str(row["segment_id"]): row for row in iter_jsonl(args.references)
        if isinstance(row.get("reference_facts"), list)
    }
    candidates: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in iter_jsonl(args.candidates):
        key = (str(row["segment_id"]), str(row.get("model_id") or row.get("extractor_model")))
        candidates.setdefault(key, []).append({
            "candidate_id": str(row.get("fact_id") or row.get("candidate_fact_id")),
            "text": str(row.get("text") or row.get("fact_text")),
        })
    rows: list[dict[str, Any]] = []
    for judgment in iter_jsonl(args.judgments):
        segment_id = str(judgment["segment_id"])
        if segment_id not in segments or segment_id not in references:
            raise ValueError(f"review inputs are missing segment: {segment_id}")
        gold_facts = [
            {
                "gold_fact_id": fact["reference_fact_id"],
                "gold_fact_text": fact.get("text") or fact.get("fact_text"),
            }
            for fact in references[segment_id]["reference_facts"]
        ]
        for result in judgment["candidate_set_results"]:
            model_id = str(result["model_id"])
            rows.append({
                "dataset_name": judgment.get("dataset_name") or judgment.get("dataset"),
                "split": judgment["split"], "sample_id": judgment["sample_id"],
                "segment_id": segment_id, "model_id": model_id,
                "segment_text": segments[segment_id],
                "gold_facts": gold_facts,
                "candidate_facts": sorted(candidates.get((segment_id, model_id), []), key=lambda item: item["candidate_id"]),
                "candidate_assessments": result["candidate_assessments"],
                "gold_fact_assessments": result["gold_fact_assessments"],
                "review_status": "", "review_notes": "",
            })
    rows.sort(key=lambda row: (row["sample_id"], row["segment_id"], row["model_id"]))
    write_jsonl(args.jsonl_output, rows)
    _write_csv(args.csv_output, rows)
    manifest = {
        "schema_version": "segment_set_pilot_review_manifest_v1",
        "review_row_count": len(rows),
        "segment_count": len({row["segment_id"] for row in rows}),
        "model_count": len({row["model_id"] for row in rows}),
        "review_scope": {"source_provenance": False, "redundancy": False, "semantic_and_temporal": True},
        "references_sha256": file_sha256(args.references),
        "candidates_sha256": file_sha256(args.candidates),
        "judgments_sha256": file_sha256(args.judgments),
        "jsonl_output": str(args.jsonl_output.resolve()),
        "jsonl_sha256": file_sha256(args.jsonl_output),
        "csv_output": str(args.csv_output.resolve()),
        "csv_sha256": file_sha256(args.csv_output),
    }
    atomic_write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    temporary.replace(path)


if __name__ == "__main__":
    main()

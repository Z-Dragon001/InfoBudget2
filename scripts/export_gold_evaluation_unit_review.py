"""Export Gold evaluation-unit Pilot rows for human review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--gold-units", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    references = {
        str(row["segment_id"]): row
        for row in iter_jsonl(args.references)
        if isinstance(row.get("reference_facts"), list)
    }
    rows: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    seen_facts: set[str] = set()
    resolution_counts: Counter[str] = Counter()
    timed_fact_count = 0

    for unit_row in iter_jsonl(args.gold_units):
        segment_id = str(unit_row["segment_id"])
        if segment_id in seen_segments:
            raise ValueError(f"duplicate Gold-unit segment: {segment_id}")
        seen_segments.add(segment_id)
        reference_row = references.get(segment_id)
        if reference_row is None:
            raise ValueError(f"Gold units reference an unknown segment: {segment_id}")
        reference_by_id = {
            str(fact["reference_fact_id"]): fact
            for fact in reference_row["reference_facts"]
        }
        returned_ids = {str(fact["gold_fact_id"]) for fact in unit_row["gold_facts"]}
        if returned_ids != set(reference_by_id):
            raise ValueError(f"Gold Fact set mismatch for {segment_id}")

        for fact in unit_row["gold_facts"]:
            fact_id = str(fact["gold_fact_id"])
            if fact_id in seen_facts:
                raise ValueError(f"duplicate Gold Fact ID: {fact_id}")
            seen_facts.add(fact_id)
            reference = reference_by_id[fact_id]
            claims = fact["claim_units"]
            required_times = [
                claim["required_time"]
                for claim in claims
                if claim["required_time"]["required"]
            ]
            if required_times:
                timed_fact_count += 1
            for required_time in required_times:
                resolution_counts[str(required_time["resolution"])] += 1
            rows.append(
                {
                    "sample_id": unit_row["sample_id"],
                    "segment_id": segment_id,
                    "gold_fact_id": fact_id,
                    "gold_fact_text": reference.get("text") or reference.get("fact_text"),
                    "claim_count": len(claims),
                    "claim_units_readable": "\n".join(
                        f"{claim['claim_id']} | {claim['claim_text']}"
                        for claim in claims
                    ),
                    "has_required_exact_time": bool(required_times),
                    "required_times_readable": "\n".join(
                        f"{item.get('surface_form')} => {item.get('normalized_value')} "
                        f"[{item.get('resolution')}]"
                        for item in required_times
                    ),
                    "claim_units_json": claims,
                    "review_status": "",
                    "corrected_claim_units_json": "",
                    "review_notes": "",
                }
            )

    rows.sort(key=lambda row: (row["sample_id"], row["segment_id"], row["gold_fact_id"]))
    write_jsonl(args.jsonl_output, rows)
    _write_csv(args.csv_output, rows)
    manifest = {
        "schema_version": "gold_evaluation_unit_review_manifest_v1",
        "review_scope": "claim decomposition and exact-time requirements only",
        "segment_count": len(seen_segments),
        "gold_fact_count": len(rows),
        "claim_unit_count": sum(int(row["claim_count"]) for row in rows),
        "timed_gold_fact_count": timed_fact_count,
        "time_resolution_counts": dict(sorted(resolution_counts.items())),
        "references_sha256": file_sha256(args.references),
        "gold_units_sha256": file_sha256(args.gold_units),
        "jsonl_output": str(args.jsonl_output.resolve()),
        "jsonl_sha256": file_sha256(args.jsonl_output),
        "csv_output": str(args.csv_output.resolve()),
        "csv_sha256": file_sha256(args.csv_output),
        "blank_review_status_means": "not_reviewed",
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
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


if __name__ == "__main__":
    main()

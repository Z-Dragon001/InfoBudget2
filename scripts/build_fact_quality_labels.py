"""Build source-grounded silver Fact-F1 labels from frozen references and Judge pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from infobudget.quality_router.io import iter_jsonl, load_capability_profiles, write_jsonl
from infobudget.quality_router.labeling import (
    build_quality_label,
    equivalence_from_pairs,
    normalized_exact_equivalence,
)
from infobudget.quality_router.schemas import AtomicFact, FactSetKey
from infobudget.rl_router.schemas import TopicSegment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--judge-decisions", type=Path)
    parser.add_argument("--allow-exact-baseline", action="store_true")
    parser.add_argument("--default-extraction-run-id", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matches-output", type=Path)
    args = parser.parse_args()
    if args.judge_decisions is None and not args.allow_exact_baseline:
        parser.error(
            "paper labels require --judge-decisions; use --allow-exact-baseline only for pipeline tests"
        )

    profiles = load_capability_profiles(args.capabilities)
    segments = _load_segments(args.segments)
    references = _load_references(args.references)
    candidates, run_ids = _load_candidates(args.candidates)
    accepted_pairs = _load_judge_decisions(args.judge_decisions) if args.judge_decisions else {}

    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for key_tuple, segment in sorted(segments.items()):
        key = FactSetKey(*key_tuple)
        if key_tuple not in references:
            raise ValueError(f"reference facts are missing for segment: {key_tuple}")
        for model_id, profile in sorted(profiles.items()):
            candidate_key = (*key_tuple, model_id)
            equivalent = (
                equivalence_from_pairs(accepted_pairs.get(candidate_key, ()))
                if args.judge_decisions
                else normalized_exact_equivalence
            )
            label, matches = build_quality_label(
                key=key,
                model_id=model_id,
                profile_id=profile.profile_id,
                candidates=candidates.get(candidate_key, ()),
                references=references[key_tuple],
                valid_source_turn_ids=set(segment.turn_ids),
                candidate_extraction_run_id=run_ids.get(
                    candidate_key, args.default_extraction_run_id
                ),
                equivalent=equivalent,
            )
            rows.append(label.to_dict())
            match_rows.append(
                {
                    **{name: value for name, value in zip(("dataset", "split", "sample_id", "segment_id"), key_tuple)},
                    "model_id": model_id,
                    "matched_pairs": [list(pair) for pair in matches.matched_pairs],
                }
            )
    write_jsonl(args.output, rows)
    if args.matches_output:
        write_jsonl(args.matches_output, match_rows)
    print(json.dumps({"labels": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))


def _load_segments(path: Path) -> dict[tuple[str, str, str, str], TopicSegment]:
    result: dict[tuple[str, str, str, str], TopicSegment] = {}
    for row in iter_jsonl(path):
        if not {"segment_id", "text", "turn_ids", "dataset_name"}.issubset(row):
            continue
        segment = TopicSegment.from_dict(row)
        key = (segment.dataset_name, segment.split, segment.sample_id, segment.segment_id)
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = segment
    if not result:
        raise ValueError("no topic segments found")
    return result


def _load_references(path: Path) -> dict[tuple[str, str, str, str], list[AtomicFact]]:
    result = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row).tuple()
        facts = [AtomicFact.from_dict(item, id_fields=("reference_fact_id", "fact_id")) for item in row.get("reference_facts", ())]
        if key in result:
            raise ValueError(f"duplicate reference fact set: {key}")
        result[key] = facts
    return result


def _load_candidates(path: Path) -> tuple[dict[tuple[str, str, str, str, str], list[AtomicFact]], dict[tuple[str, str, str, str, str], str]]:
    grouped: dict[tuple[str, str, str, str, str], list[AtomicFact]] = defaultdict(list)
    run_ids: dict[tuple[str, str, str, str, str], str] = {}
    for row in _artifact_rows(path, "memories"):
        key = FactSetKey.from_dict(row).tuple()
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        if not model_id:
            raise ValueError("candidate row is missing extractor_model/model_id")
        group_key = (*key, model_id)
        grouped[group_key].append(AtomicFact.from_dict(row, id_fields=("fact_id", "candidate_fact_id")))
        run_id = str(row.get("candidate_extraction_run_id") or row.get("extraction_run_id") or "").strip()
        if run_id:
            previous = run_ids.setdefault(group_key, run_id)
            if previous != run_id:
                raise ValueError(f"candidate segment/model mixes extraction runs: {group_key}")
    return dict(grouped), run_ids


def _load_judge_decisions(path: Path) -> dict[tuple[str, str, str, str, str], set[tuple[str, str]]]:
    grouped: dict[tuple[str, str, str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in iter_jsonl(path):
        if not bool(row.get("equivalent")):
            continue
        key = (*FactSetKey.from_dict(row).tuple(), str(row.get("model_id") or row.get("extractor_model") or ""))
        grouped[key].add((str(row["candidate_fact_id"]), str(row["reference_fact_id"])))
    return dict(grouped)


def _artifact_rows(path: Path, list_field: str) -> Iterable[dict[str, Any]]:
    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get(list_field, ()):
            yield row
        return
    yield from iter_jsonl(path)


if __name__ == "__main__":
    main()

"""Compute complete set metrics from candidates, frozen references, and Judge pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from infobudget.quality_router.io import iter_jsonl, write_jsonl
from infobudget.quality_router.schemas import AtomicFact, FactSetKey
from reference_fact_pipeline.metrics import score_fact_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judge-decisions", type=Path)
    parser.add_argument("--allow-exact-baseline", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.judge_decisions is None and not args.allow_exact_baseline:
        parser.error("provide --judge-decisions; exact matching is only a diagnostic baseline")

    references = _load_references(args.references)
    candidates = _load_candidates(args.candidates)
    pairs = _load_pairs(args.judge_decisions) if args.judge_decisions else {}
    rows: list[dict[str, Any]] = []
    for candidate_key, facts in sorted(candidates.items()):
        fact_key = candidate_key[:4]
        if fact_key not in references:
            raise ValueError(f"missing reference Fact set for {fact_key}")
        reference_facts, valid_source_ids, reference_hash = references[fact_key]
        invalid_ids = {
            fact.fact_id
            for fact in facts
            if not fact.source_turn_ids or not set(fact.source_turn_ids).issubset(valid_source_ids)
        }
        if args.judge_decisions:
            candidate_by_id = {item.fact_id: item for item in facts}
            reference_by_id = {item.fact_id: item for item in reference_facts}
            equivalent_pairs = {
                (candidate_id, reference_id)
                for candidate_id, reference_id in pairs.get(candidate_key, set())
                if candidate_id in candidate_by_id
                and reference_id in reference_by_id
                and bool(
                    set(candidate_by_id[candidate_id].source_turn_ids)
                    & set(reference_by_id[reference_id].source_turn_ids)
                )
            }
        else:
            equivalent_pairs = {
                (candidate.fact_id, reference.fact_id)
                for candidate in facts
                for reference in reference_facts
                if _normalize(candidate.text) == _normalize(reference.text)
                and bool(set(candidate.source_turn_ids) & set(reference.source_turn_ids))
            }
        metrics = score_fact_sets(
            (item.fact_id for item in facts),
            (item.fact_id for item in reference_facts),
            equivalent_pairs,
            invalid_candidate_ids=invalid_ids,
            beta=args.beta,
        )
        rows.append(
            {
                **dict(zip(("dataset", "split", "sample_id", "segment_id", "model_id"), candidate_key)),
                "reference_set_hash": reference_hash,
                "metric_version": "fact_set_metrics_v1",
                **metrics.to_dict(),
            }
        )
    write_jsonl(args.output, rows)
    print(json.dumps({"metric_rows": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))


def _load_references(
    path: Path,
) -> dict[tuple[str, str, str, str], tuple[list[AtomicFact], set[int], str]]:
    result = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row).tuple()
        if key in result:
            raise ValueError(f"duplicate reference set: {key}")
        facts = [
            AtomicFact.from_dict(item, id_fields=("reference_fact_id", "fact_id"))
            for item in row.get("reference_facts", ())
        ]
        result[key] = (
            facts,
            {int(item) for item in row.get("segment_turn_ids", ())},
            str(row.get("reference_set_hash") or ""),
        )
    return result


def _load_candidates(path: Path) -> dict[tuple[str, str, str, str, str], list[AtomicFact]]:
    result: dict[tuple[str, str, str, str, str], list[AtomicFact]] = defaultdict(list)
    for row in _artifact_rows(path, "memories"):
        key = FactSetKey.from_dict(row).tuple()
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        if not model_id:
            raise ValueError("candidate row lacks model_id/extractor_model")
        result[(*key, model_id)].append(
            AtomicFact.from_dict(row, id_fields=("fact_id", "candidate_fact_id"))
        )
    return dict(result)


def _load_pairs(path: Path) -> dict[tuple[str, str, str, str, str], set[tuple[str, str]]]:
    result: dict[tuple[str, str, str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in iter_jsonl(path):
        if not bool(row.get("equivalent")):
            continue
        key = (
            *FactSetKey.from_dict(row).tuple(),
            str(row.get("model_id") or row.get("extractor_model") or ""),
        )
        result[key].add((str(row["candidate_fact_id"]), str(row["reference_fact_id"])))
    return dict(result)


def _artifact_rows(path: Path, list_field: str) -> Iterable[dict[str, Any]]:
    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from payload.get(list_field, ())
        return
    yield from iter_jsonl(path)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


if __name__ == "__main__":
    main()

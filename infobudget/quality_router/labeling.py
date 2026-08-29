"""Build one scalar, source-grounded silver Fact-F1 label per segment/model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from infobudget.quality_router.schemas import (
    AtomicFact,
    FactQualityLabel,
    FactSetKey,
)

FactEquivalence = Callable[[AtomicFact, AtomicFact], bool]


@dataclass(frozen=True, slots=True)
class MatchResult:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    matched_pairs: tuple[tuple[str, str], ...]


def normalized_exact_equivalence(candidate: AtomicFact, reference: AtomicFact) -> bool:
    """Deterministic baseline; paper experiments should pass frozen Judge decisions."""
    return _normalize(candidate.text) == _normalize(reference.text)


def score_fact_sets(
    candidates: Iterable[AtomicFact],
    references: Iterable[AtomicFact],
    *,
    valid_source_turn_ids: set[int],
    equivalent: FactEquivalence = normalized_exact_equivalence,
) -> MatchResult:
    candidate_list = list(candidates)
    reference_list = list(references)
    _require_unique_ids(candidate_list, "candidate")
    _require_unique_ids(reference_list, "reference")

    valid_candidates = [
        fact
        for fact in candidate_list
        if fact.source_turn_ids
        and set(fact.source_turn_ids).issubset(valid_source_turn_ids)
    ]
    adjacency = [
        [
            index
            for index, reference in enumerate(reference_list)
            if equivalent(candidate, reference)
            and bool(set(candidate.source_turn_ids) & set(reference.source_turn_ids))
        ]
        for candidate in valid_candidates
    ]
    matched_reference_to_candidate: dict[int, int] = {}

    def augment(candidate_index: int, seen: set[int]) -> bool:
        for reference_index in adjacency[candidate_index]:
            if reference_index in seen:
                continue
            seen.add(reference_index)
            previous = matched_reference_to_candidate.get(reference_index)
            if previous is None or augment(previous, seen):
                matched_reference_to_candidate[reference_index] = candidate_index
                return True
        return False

    for candidate_index in range(len(valid_candidates)):
        augment(candidate_index, set())

    pairs = tuple(
        sorted(
            (
                valid_candidates[candidate_index].fact_id,
                reference_list[reference_index].fact_id,
            )
            for reference_index, candidate_index in matched_reference_to_candidate.items()
        )
    )
    tp = len(pairs)
    fp = len(candidate_list) - tp
    fn = len(reference_list) - tp
    if not candidate_list and not reference_list:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return MatchResult(tp, fp, fn, precision, recall, f1, pairs)


def build_quality_label(
    *,
    key: FactSetKey,
    model_id: str,
    profile_id: str,
    candidates: Iterable[AtomicFact],
    references: Iterable[AtomicFact],
    valid_source_turn_ids: set[int],
    candidate_extraction_run_id: str,
    equivalent: FactEquivalence = normalized_exact_equivalence,
    label_version: str = "silver_f1_v1",
) -> tuple[FactQualityLabel, MatchResult]:
    reference_list = list(references)
    result = score_fact_sets(
        candidates,
        reference_list,
        valid_source_turn_ids=valid_source_turn_ids,
        equivalent=equivalent,
    )
    reference_set_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "fact_id": item.fact_id,
                    "text": item.text,
                    "source_turn_ids": item.source_turn_ids,
                }
                for item in sorted(reference_list, key=lambda value: value.fact_id)
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        FactQualityLabel(
            key=key,
            model_id=model_id,
            profile_id=profile_id,
            true_positive=result.true_positive,
            false_positive=result.false_positive,
            false_negative=result.false_negative,
            precision=result.precision,
            recall=result.recall,
            silver_strict_fact_f1=result.f1,
            reference_set_hash=reference_set_hash,
            candidate_extraction_run_id=candidate_extraction_run_id,
            label_version=label_version,
        ),
        result,
    )


def equivalence_from_pairs(
    equivalent_pairs: Iterable[tuple[str, str]],
) -> FactEquivalence:
    """Create a predicate from frozen candidate/reference Judge decisions."""
    accepted = set(equivalent_pairs)
    return lambda candidate, reference: (candidate.fact_id, reference.fact_id) in accepted


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


def _require_unique_ids(facts: list[AtomicFact], label: str) -> None:
    ids = [fact.fact_id for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {label} fact_id")

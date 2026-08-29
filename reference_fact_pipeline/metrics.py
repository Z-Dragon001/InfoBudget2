"""Set-level Fact metrics with explicit formula implementations.

No true-negative universe exists for open-ended Fact extraction, so accuracy and
specificity are intentionally not reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class FactSetMetrics:
    candidate_count: int
    reference_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f_beta: float
    beta: float
    jaccard: float
    false_discovery_rate: float
    false_negative_rate: float
    exact_set_match: float
    source_validity_rate: float
    matched_pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["matched_pairs"] = [list(item) for item in self.matched_pairs]
        return value


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    *,
    beta: float = 1.0,
    candidate_count: int | None = None,
    reference_count: int | None = None,
    valid_candidate_count: int | None = None,
    matched_pairs: Iterable[tuple[str, str]] = (),
) -> FactSetMetrics:
    """Compute P, R, F1, F-beta, Jaccard, FDR, FNR and exact set match.

    Formulae: P=TP/(TP+FP), R=TP/(TP+FN),
    F1=2TP/(2TP+FP+FN), F_beta=(1+b²)TP/((1+b²)TP+b²FN+FP),
    J=TP/(TP+FP+FN), FDR=FP/(TP+FP), FNR=FN/(TP+FN).

    Empty-set convention: comparing two empty sets is a perfect match. If only
    the reference set is empty, recall is 1 (nothing was omitted) but precision
    and F scores are 0 because every prediction is false.
    """
    if min(tp, fp, fn) < 0:
        raise ValueError("tp, fp and fn must be non-negative")
    if beta <= 0:
        raise ValueError("beta must be positive")
    candidates = tp + fp if candidate_count is None else int(candidate_count)
    references = tp + fn if reference_count is None else int(reference_count)
    if candidates < tp + fp or references != tp + fn:
        raise ValueError("counts are inconsistent with set sizes")
    if candidates == 0 and references == 0:
        precision = recall = f1 = f_beta = jaccard = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
        beta_sq = beta * beta
        f_beta_denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
        f_beta = (
            (1 + beta_sq) * tp / f_beta_denominator
            if f_beta_denominator
            else 0.0
        )
        union = tp + fp + fn
        jaccard = tp / union if union else 1.0
    false_discovery_rate = fp / (tp + fp) if tp + fp else 0.0
    false_negative_rate = fn / (tp + fn) if tp + fn else 0.0
    valid = candidates if valid_candidate_count is None else int(valid_candidate_count)
    if not 0 <= valid <= candidates:
        raise ValueError("valid_candidate_count must lie in [0, candidate_count]")
    source_validity_rate = valid / candidates if candidates else 1.0
    return FactSetMetrics(
        candidate_count=candidates,
        reference_count=references,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        f_beta=f_beta,
        beta=float(beta),
        jaccard=jaccard,
        false_discovery_rate=false_discovery_rate,
        false_negative_rate=false_negative_rate,
        exact_set_match=float(fp == 0 and fn == 0),
        source_validity_rate=source_validity_rate,
        matched_pairs=tuple(matched_pairs),
    )


def score_fact_sets(
    candidate_ids: Iterable[str],
    reference_ids: Iterable[str],
    equivalent_pairs: Iterable[tuple[str, str]],
    *,
    invalid_candidate_ids: Iterable[str] = (),
    beta: float = 1.0,
) -> FactSetMetrics:
    """Use maximum one-to-one bipartite matching, preventing duplicate TP credit."""
    candidates = tuple(str(item) for item in candidate_ids)
    references = tuple(str(item) for item in reference_ids)
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate Fact IDs must be unique")
    if len(references) != len(set(references)):
        raise ValueError("reference Fact IDs must be unique")
    candidate_set = set(candidates)
    reference_set = set(references)
    invalid = set(str(item) for item in invalid_candidate_ids) & candidate_set
    adjacency: dict[str, list[str]] = {item: [] for item in candidates if item not in invalid}
    for candidate_id, reference_id in equivalent_pairs:
        if candidate_id in adjacency and reference_id in reference_set:
            adjacency[candidate_id].append(reference_id)
    for values in adjacency.values():
        values.sort()

    reference_to_candidate: dict[str, str] = {}

    def augment(candidate_id: str, visited: set[str]) -> bool:
        for reference_id in adjacency[candidate_id]:
            if reference_id in visited:
                continue
            visited.add(reference_id)
            owner = reference_to_candidate.get(reference_id)
            if owner is None or augment(owner, visited):
                reference_to_candidate[reference_id] = candidate_id
                return True
        return False

    for candidate_id in sorted(adjacency):
        augment(candidate_id, set())
    pairs = tuple(
        sorted((candidate_id, reference_id) for reference_id, candidate_id in reference_to_candidate.items())
    )
    tp = len(pairs)
    return metrics_from_counts(
        tp,
        len(candidates) - tp,
        len(references) - tp,
        beta=beta,
        candidate_count=len(candidates),
        reference_count=len(references),
        valid_candidate_count=len(candidates) - len(invalid),
        matched_pairs=pairs,
    )

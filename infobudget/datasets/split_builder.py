"""Deterministic grouped-stratified split construction for LongMemEval."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitFeature:
    sample_id: str
    question_type: str
    is_unanswerable: bool
    segment_bin: str
    evidence_session_ids: tuple[str, ...]
    session_ids: tuple[str, ...]

    @property
    def stratum(self) -> tuple[str, bool, str]:
        return (self.question_type, self.is_unanswerable, self.segment_bin)


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    samples: tuple[SplitFeature, ...]

    @property
    def size(self) -> int:
        return len(self.samples)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def load_longmemeval_features(
    root: str | Path,
    *,
    source_split: str = "full",
    segmentation_method: str = "nsp_text_tiling",
) -> tuple[list[SplitFeature], list[int]]:
    """Load leakage-grouping fields plus frozen segment-count quartile bins."""

    root = Path(root).resolve()
    processed = root / "datasets" / "processed" / "longmemeval" / source_split
    questions = _read_jsonl(processed / "questions.jsonl")
    sessions_by_sample: dict[str, list[str]] = defaultdict(list)
    for row in _read_jsonl(processed / "sessions.jsonl"):
        sessions_by_sample[str(row["sample_id"])].append(str(row["session_id"]))

    segment_root = (
        root
        / "datasets"
        / "segmented"
        / "longmemeval"
        / source_split
        / segmentation_method
        / "samples"
    )
    segment_counts = {
        item.name: _nonempty_line_count(item / "segments.jsonl")
        for item in segment_root.iterdir()
        if item.is_dir() and (item / "segments.jsonl").is_file()
    }
    sample_ids = {str(row["sample_id"]) for row in questions}
    if len(questions) != 500 or len(sample_ids) != 500:
        raise ValueError("LongMemEval split construction requires exactly 500 unique samples")
    if set(segment_counts) != sample_ids or set(sessions_by_sample) != sample_ids:
        raise ValueError("processed questions, sessions, and segmented samples do not have identical coverage")

    ordered_counts = sorted(segment_counts.values())
    thresholds = [ordered_counts[index] for index in (124, 249, 374)]
    features = [
        SplitFeature(
            sample_id=str(row["sample_id"]),
            question_type=str(row["question_type"]),
            is_unanswerable=bool(row["is_unanswerable"]),
            segment_bin=_segment_bin(segment_counts[str(row["sample_id"])], thresholds),
            evidence_session_ids=tuple(sorted(str(value) for value in row["evidence_session_ids"])),
            session_ids=tuple(sorted(sessions_by_sample[str(row["sample_id"])])),
        )
        for row in questions
    ]
    return sorted(features, key=lambda item: item.sample_id), thresholds


def build_evidence_groups(features: list[SplitFeature]) -> list[EvidenceGroup]:
    """Group every occurrence of global evidence sessions and abstention counterparts."""

    by_id = {feature.sample_id: feature for feature in features}
    union_find = _UnionFind(by_id)
    evidence_universe = {
        session_id for feature in features for session_id in feature.evidence_session_ids
    }
    samples_by_protected_session: dict[str, list[str]] = defaultdict(list)
    for feature in features:
        for session_id in feature.session_ids:
            if session_id in evidence_universe:
                samples_by_protected_session[session_id].append(feature.sample_id)
    for sample_ids in samples_by_protected_session.values():
        for sample_id in sample_ids[1:]:
            union_find.union(sample_ids[0], sample_id)
    for sample_id in by_id:
        counterpart = sample_id.removesuffix("_abs")
        if counterpart in by_id:
            union_find.union(sample_id, counterpart)

    members: dict[str, list[SplitFeature]] = defaultdict(list)
    for sample_id, feature in by_id.items():
        members[union_find.find(sample_id)].append(feature)
    return [
        EvidenceGroup(min(item.sample_id for item in values), tuple(sorted(values, key=lambda item: item.sample_id)))
        for values in members.values()
    ]


def grouped_stratified_assignment(
    groups: list[EvidenceGroup],
    target_sizes: dict[str, int],
    *,
    seed: int,
    unanswerable_targets: dict[str, int] | None = None,
) -> dict[str, list[EvidenceGroup]]:
    """Greedily minimize joint-stratum error while exactly filling partition sizes."""

    total = sum(group.size for group in groups)
    if sum(target_sizes.values()) != total:
        raise ValueError("partition target sizes do not match the number of samples")
    global_counts = _balance_counts(sample for group in groups for sample in group.samples)
    targets = {
        partition: {
            key: value * size / total
            for key, value in global_counts.items()
        }
        for partition, size in target_sizes.items()
    }
    rarity = Counter(sample.stratum for group in groups for sample in group.samples)
    ordered = sorted(
        groups,
        key=lambda group: (
            -group.size,
            -sum(1.0 / rarity[sample.stratum] for sample in group.samples),
            _stable_tie(seed, group.group_id, "order"),
        ),
    )
    assigned = {partition: [] for partition in target_sizes}
    sizes = Counter()
    unanswerable_sizes = Counter()
    counts = {partition: Counter() for partition in target_sizes}
    for group in ordered:
        group_counts = _balance_counts(group.samples)
        group_unanswerable = sum(sample.is_unanswerable for sample in group.samples)
        choices = []
        for partition, capacity in target_sizes.items():
            if sizes[partition] + group.size > capacity:
                continue
            if (
                unanswerable_targets is not None
                and unanswerable_sizes[partition] + group_unanswerable
                > unanswerable_targets[partition]
            ):
                continue
            delta = 0.0
            for key, increment in group_counts.items():
                target = targets[partition][key]
                before = counts[partition][key] - target
                after = before + increment
                delta += (after * after - before * before) / max(target, 1.0)
            choices.append((delta, _stable_tie(seed, group.group_id, partition), partition))
        if not choices:
            raise ValueError(f"cannot place evidence group {group.group_id} without exceeding target sizes")
        partition = min(choices)[2]
        assigned[partition].append(group)
        sizes[partition] += group.size
        unanswerable_sizes[partition] += group_unanswerable
        counts[partition].update(group_counts)
    _refine_equal_size_swaps(assigned, counts, targets, seed=seed)
    if dict(sizes) != target_sizes:
        raise ValueError(f"group assignment failed to fill exact targets: {dict(sizes)}")
    if unanswerable_targets is not None and any(
        unanswerable_sizes[partition] != target
        for partition, target in unanswerable_targets.items()
    ):
        raise ValueError(
            f"group assignment failed to fill unanswerable targets: {dict(unanswerable_sizes)}"
        )
    return assigned


def _refine_equal_size_swaps(assigned, counts, targets, *, seed: int) -> None:
    """Improve soft strata without changing sample or unanswerable hard quotas."""

    partitions = list(assigned)
    for iteration in range(30):
        best = None
        for left_index, left in enumerate(partitions):
            for right in partitions[left_index + 1 :]:
                for left_position, left_group in enumerate(assigned[left]):
                    left_unanswerable = sum(sample.is_unanswerable for sample in left_group.samples)
                    left_counts = _balance_counts(left_group.samples)
                    for right_position, right_group in enumerate(assigned[right]):
                        if left_group.size != right_group.size:
                            continue
                        if left_unanswerable != sum(
                            sample.is_unanswerable for sample in right_group.samples
                        ):
                            continue
                        right_counts = _balance_counts(right_group.samples)
                        keys = set(left_counts) | set(right_counts)
                        delta = 0.0
                        for key in keys:
                            left_before = counts[left][key] - targets[left][key]
                            right_before = counts[right][key] - targets[right][key]
                            left_after = left_before - left_counts[key] + right_counts[key]
                            right_after = right_before - right_counts[key] + left_counts[key]
                            delta += (
                                left_after * left_after
                                - left_before * left_before
                            ) / max(targets[left][key], 1.0)
                            delta += (
                                right_after * right_after
                                - right_before * right_before
                            ) / max(targets[right][key], 1.0)
                        candidate = (
                            delta,
                            _stable_tie(
                                seed + iteration,
                                left_group.group_id,
                                right_group.group_id,
                                "swap",
                            ),
                            left,
                            right,
                            left_position,
                            right_position,
                            left_counts,
                            right_counts,
                        )
                        if delta < -1e-9 and (best is None or candidate[:2] < best[:2]):
                            best = candidate
        if best is None:
            break
        _, _, left, right, left_position, right_position, left_counts, right_counts = best
        assigned[left][left_position], assigned[right][right_position] = (
            assigned[right][right_position],
            assigned[left][left_position],
        )
        counts[left].subtract(left_counts)
        counts[left].update(right_counts)
        counts[right].subtract(right_counts)
        counts[right].update(left_counts)


def build_longmemeval_manifests(
    root: str | Path,
    *,
    segmentation_method: str = "nsp_text_tiling",
    seed: int = 42,
) -> tuple[dict, dict]:
    """Build fixed 400/50/50 and nested five-fold 360/40/100 manifests."""

    root = Path(root).resolve()
    features, thresholds = load_longmemeval_features(
        root,
        segmentation_method=segmentation_method,
    )
    groups = build_evidence_groups(features)
    common = _common_manifest_fields(root, features, groups, thresholds, segmentation_method, seed)

    fixed_type_targets = {
        "train": {
            "knowledge-update": 63,
            "multi-session": 106,
            "single-session-assistant": 45,
            "single-session-preference": 24,
            "single-session-user": 56,
            "temporal-reasoning": 106,
        },
        "validation": {
            "knowledge-update": 8,
            "multi-session": 13,
            "single-session-assistant": 6,
            "single-session-preference": 3,
            "single-session-user": 7,
            "temporal-reasoning": 13,
        },
        "test": {
            "knowledge-update": 7,
            "multi-session": 14,
            "single-session-assistant": 5,
            "single-session-preference": 3,
            "single-session-user": 7,
            "temporal-reasoning": 14,
        },
    }
    fixed_unanswerable_targets = _allocate_type_targets(
        Counter(sample.question_type for sample in features if sample.is_unanswerable),
        {"train": 24, "validation": 3, "test": 3},
        seed=seed,
    )
    fixed_assignment = _assign_by_question_type(
        groups,
        fixed_type_targets,
        fixed_unanswerable_targets,
        seed=seed,
    )
    fixed_fold = _fold_payload(0, fixed_assignment, features)
    fixed = {**common, "protocol": "fixed_80_10_10", "folds": [fixed_fold]}

    type_counts = Counter(sample.question_type for sample in features)
    outer_type_targets = _allocate_type_targets(
        type_counts,
        {f"outer_{fold}": 100 for fold in range(5)},
        seed=seed,
    )
    outer_unanswerable_targets = _allocate_type_targets(
        Counter(sample.question_type for sample in features if sample.is_unanswerable),
        {f"outer_{fold}": 6 for fold in range(5)},
        seed=seed,
    )
    outer = _assign_by_question_type(
        groups,
        outer_type_targets,
        outer_unanswerable_targets,
        seed=seed,
    )
    cv_folds = []
    for fold in range(5):
        test_groups = outer[f"outer_{fold}"]
        remaining = [group for index in range(5) if index != fold for group in outer[f"outer_{index}"]]
        remaining_type_counts = Counter(
            sample.question_type for group in remaining for sample in group.samples
        )
        inner_type_targets = _allocate_type_targets(
            remaining_type_counts,
            {"train": 360, "validation": 40},
            seed=seed + fold + 1,
        )
        inner_unanswerable_targets = _allocate_type_targets(
            Counter(
                sample.question_type
                for group in remaining
                for sample in group.samples
                if sample.is_unanswerable
            ),
            {"train": 22, "validation": 2},
            seed=seed + fold + 1,
        )
        inner = _assign_by_question_type(
            remaining,
            inner_type_targets,
            inner_unanswerable_targets,
            seed=seed + fold + 1,
        )
        cv_folds.append(
            _fold_payload(
                fold,
                {"train": inner["train"], "validation": inner["validation"], "test": test_groups},
                features,
            )
        )
    cv5 = {**common, "protocol": "cv5_nested_360_40_100", "folds": cv_folds}
    return fixed, cv5


def _assign_by_question_type(
    groups: list[EvidenceGroup],
    targets: dict[str, dict[str, int]],
    unanswerable_targets: dict[str, dict[str, int]],
    *,
    seed: int,
) -> dict[str, list[EvidenceGroup]]:
    groups_by_type: dict[str, list[EvidenceGroup]] = defaultdict(list)
    for group in groups:
        question_types = {sample.question_type for sample in group.samples}
        if len(question_types) != 1:
            raise ValueError(f"evidence group {group.group_id} mixes question types")
        groups_by_type[next(iter(question_types))].append(group)
    assigned = {partition: [] for partition in targets}
    for question_type, typed_groups in sorted(groups_by_type.items()):
        typed_targets = {
            partition: values.get(question_type, 0)
            for partition, values in targets.items()
        }
        typed_assignment = grouped_stratified_assignment(
            typed_groups,
            typed_targets,
            seed=seed + int(_stable_tie(seed, question_type, "typed")[:8], 16),
            unanswerable_targets={
                partition: values.get(question_type, 0)
                for partition, values in unanswerable_targets.items()
            },
        )
        for partition, values in typed_assignment.items():
            assigned[partition].extend(values)
    return assigned


def _allocate_type_targets(
    type_counts: Counter,
    partition_sizes: dict[str, int],
    *,
    seed: int,
) -> dict[str, dict[str, int]]:
    total = sum(type_counts.values())
    if total != sum(partition_sizes.values()):
        raise ValueError("type counts and partition sizes have different totals")
    targets = {partition: {} for partition in partition_sizes}
    row_remaining = {}
    column_remaining = dict(partition_sizes)
    fractions = {}
    for question_type, count in type_counts.items():
        allocated = 0
        for partition, size in partition_sizes.items():
            exact = count * size / total
            base = int(exact)
            targets[partition][question_type] = base
            fractions[(question_type, partition)] = exact - base
            allocated += base
            column_remaining[partition] -= base
        row_remaining[question_type] = count - allocated
    while any(row_remaining.values()):
        choices = [
            (
                -fractions[(question_type, partition)],
                _stable_tie(seed, question_type, partition, "rounding"),
                question_type,
                partition,
            )
            for question_type, remaining in row_remaining.items()
            for partition, capacity in column_remaining.items()
            if remaining > 0 and capacity > 0
        ]
        if not choices:
            raise ValueError("cannot round question-type targets to the requested partition sizes")
        _, _, question_type, partition = min(choices)
        targets[partition][question_type] += 1
        row_remaining[question_type] -= 1
        column_remaining[partition] -= 1
    if any(column_remaining.values()):
        raise ValueError("rounded question-type targets did not fill every partition")
    return targets


def _common_manifest_fields(root, features, groups, thresholds, method, seed) -> dict:
    processed_manifest = root / "datasets" / "processed" / "longmemeval" / "full" / "manifest.json"
    segmentation_manifest = (
        root / "datasets" / "segmented" / "longmemeval" / "full" / method / "manifest.json"
    )
    return {
        "schema_version": "conversation_split_v1",
        "dataset_name": "longmemeval",
        "source_split": "full",
        "seed": seed,
        "unit": "question_history_sample",
        "source_processed_manifest_sha256": _sha256(processed_manifest),
        "stratification": {
            "fields": ["question_type", "is_unanswerable", "segment_count_bin"],
            "segment_count_reference_method": method,
            "segment_count_quartile_thresholds": thresholds,
            "source_segmentation_manifest_sha256": _sha256(segmentation_manifest),
        },
        "grouping": {
            "protected": [
                "all sample occurrences of every globally evidence-bearing session",
                "answerable/abstention counterpart IDs",
            ],
            "num_groups": len(groups),
            "group_size_counts": {
                str(size): count for size, count in sorted(Counter(group.size for group in groups).items())
            },
            "max_group_size": max(group.size for group in groups),
            "background_distractor_sessions_may_cross_partitions": True,
        },
        "dataset_statistics": _partition_statistics(features),
    }


def _fold_payload(fold, assignments, all_features) -> dict:
    partitions = {
        partition: sorted(sample.sample_id for group in groups for sample in group.samples)
        for partition, groups in assignments.items()
    }
    by_id = {sample.sample_id: sample for sample in all_features}
    payload = {"fold": fold, **partitions}
    payload["statistics"] = {
        partition: _partition_statistics([by_id[sample_id] for sample_id in sample_ids])
        for partition, sample_ids in partitions.items()
    }
    payload["audit"] = _fold_overlap_audit(assignments)
    return payload


def _fold_overlap_audit(assignments) -> dict:
    evidence_universe = {
        session_id
        for groups in assignments.values()
        for group in groups
        for sample in group.samples
        for session_id in sample.evidence_session_ids
    }
    protected = {
        partition: {
            session_id
            for group in groups
            for sample in group.samples
            for session_id in sample.session_ids
            if session_id in evidence_universe
        }
        for partition, groups in assignments.items()
    }
    distractors = {
        partition: {
            session_id
            for group in groups
            for sample in group.samples
            for session_id in sample.session_ids
            if session_id not in evidence_universe
        }
        for partition, groups in assignments.items()
    }
    partitions = list(assignments)
    evidence_overlap = {}
    distractor_overlap = {}
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            key = f"{left}__{right}"
            evidence_overlap[key] = len(protected[left] & protected[right])
            distractor_overlap[key] = len(distractors[left] & distractors[right])
    return {
        "cross_partition_evidence_session_overlap": evidence_overlap,
        "cross_partition_background_distractor_session_overlap": distractor_overlap,
    }


def _partition_statistics(features: list[SplitFeature]) -> dict:
    return {
        "num_samples": len(features),
        "question_type": dict(sorted(Counter(item.question_type for item in features).items())),
        "is_unanswerable": dict(sorted(Counter(str(item.is_unanswerable).lower() for item in features).items())),
        "segment_count_bin": dict(sorted(Counter(item.segment_bin for item in features).items())),
    }


def _balance_counts(features: Iterable[SplitFeature]) -> Counter:
    counts = Counter()
    for feature in features:
        counts[("joint", *feature.stratum)] += 2
        counts[("question_type", feature.question_type)] += 12
        counts[("is_unanswerable", feature.is_unanswerable)] += (
            12 if feature.is_unanswerable else 1
        )
        counts[("segment_bin", feature.segment_bin)] += 2
    return counts


def _segment_bin(value: int, thresholds: list[int]) -> str:
    if value <= thresholds[0]:
        return f"q1_le_{thresholds[0]}"
    if value <= thresholds[1]:
        return f"q2_{thresholds[0] + 1}_{thresholds[1]}"
    if value <= thresholds[2]:
        return f"q3_{thresholds[1] + 1}_{thresholds[2]}"
    return f"q4_ge_{thresholds[2] + 1}"


def _stable_tie(seed: int, *values: str) -> str:
    return hashlib.sha256("|".join((str(seed), *values)).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

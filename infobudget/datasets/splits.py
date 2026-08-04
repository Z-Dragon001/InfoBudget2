"""Leakage-safe conversation-level experiment split manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "conversation_split_v1"
PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSelection:
    """One selected fold from a versioned split manifest."""

    path: Path
    sha256: str
    dataset_name: str
    source_split: str
    protocol: str
    seed: int
    fold: int
    source_processed_manifest_sha256: str
    stratification_segment_method: str
    source_segmentation_manifest_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]

    def sample_ids(self, partition: str) -> tuple[str, ...]:
        if partition not in PARTITIONS:
            raise ValueError(f"unknown split partition: {partition}")
        return getattr(self, f"{partition}_sample_ids")


def load_split_selection(
    path: str | Path,
    *,
    dataset_name: str,
    source_split: str,
    fold: int,
    available_sample_ids: set[str] | None = None,
    source_processed_manifest_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> SplitSelection:
    """Load one fold and reject overlaps, duplicates, or stale sample IDs."""

    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"split manifest must use schema_version={SCHEMA_VERSION}")
    if payload.get("dataset_name") != dataset_name:
        raise ValueError(
            f"split manifest dataset mismatch: expected {dataset_name}, got {payload.get('dataset_name')}"
        )
    if payload.get("source_split") != source_split:
        raise ValueError(
            f"split manifest source_split mismatch: expected {source_split}, got {payload.get('source_split')}"
        )
    expected_source_hash = str(payload.get("source_processed_manifest_sha256") or "")
    if source_processed_manifest_path is not None:
        if not expected_source_hash:
            raise ValueError("split manifest is missing source_processed_manifest_sha256")
        actual_source_hash = hashlib.sha256(Path(source_processed_manifest_path).read_bytes()).hexdigest()
        if actual_source_hash != expected_source_hash:
            raise ValueError(
                "split manifest source_processed_manifest_sha256 does not match the processed manifest"
            )
    stratification = payload.get("stratification") or {}
    reference_method = str(stratification.get("segment_count_reference_method") or "")
    expected_segmentation_hash = str(
        stratification.get("source_segmentation_manifest_sha256") or ""
    )
    if dataset_name == "longmemeval":
        if not reference_method or not expected_segmentation_hash:
            raise ValueError(
                "LongMemEval split manifest must identify its stratification segmentation manifest"
            )
        if project_root is not None:
            segmentation_manifest = (
                Path(project_root)
                / "datasets"
                / "segmented"
                / dataset_name
                / source_split
                / reference_method
                / "manifest.json"
            )
            actual_segmentation_hash = hashlib.sha256(segmentation_manifest.read_bytes()).hexdigest()
            if actual_segmentation_hash != expected_segmentation_hash:
                raise ValueError(
                    "split manifest source_segmentation_manifest_sha256 does not match its reference manifest"
                )

    protocol = str(payload.get("protocol") or "")
    if not protocol:
        raise ValueError("split manifest protocol must be non-empty")
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("split manifest folds must be a non-empty list")
    fold_ids = [item.get("fold") for item in folds if isinstance(item, dict)]
    if len(fold_ids) != len(folds) or any(not isinstance(item, int) for item in fold_ids):
        raise ValueError("every split manifest fold must have an integer fold ID")
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("split manifest contains duplicate fold IDs")
    matching = [item for item in folds if isinstance(item, dict) and item.get("fold") == fold]
    if len(matching) != 1:
        raise ValueError(f"split manifest must contain exactly one fold={fold}")
    selected = matching[0]

    validated = {
        item["fold"]: _validate_fold(item, available_sample_ids=available_sample_ids)
        for item in folds
    }
    if protocol == "cv5":
        _validate_cv5(validated)
    elif protocol == "fixed_80_10_10":
        _validate_longmemeval_fixed(dataset_name, validated)
    elif protocol == "cv5_nested_360_40_100":
        _validate_longmemeval_cv5(dataset_name, validated)
    values = validated[fold]
    return SplitSelection(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        dataset_name=dataset_name,
        source_split=source_split,
        protocol=protocol,
        seed=int(payload["seed"]),
        fold=fold,
        source_processed_manifest_sha256=expected_source_hash,
        stratification_segment_method=reference_method,
        source_segmentation_manifest_sha256=expected_segmentation_hash,
        train_sample_ids=values["train"],
        validation_sample_ids=values["validation"],
        test_sample_ids=values["test"],
    )


def _validate_fold(
    value: dict,
    *,
    available_sample_ids: set[str] | None,
) -> dict[str, tuple[str, ...]]:
    fold = value["fold"]
    partitions: dict[str, tuple[str, ...]] = {}
    for partition in PARTITIONS:
        items = value.get(partition, [])
        if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
            raise ValueError(f"fold {fold} partition {partition} must be a list of sample IDs")
        if len(items) != len(set(items)):
            raise ValueError(f"fold {fold} partition {partition} contains duplicate sample IDs")
        partitions[partition] = tuple(items)
    if not partitions["train"]:
        raise ValueError(f"fold {fold} has an empty training partition")
    if not partitions["test"]:
        raise ValueError(f"fold {fold} has an empty test partition")

    memberships: dict[str, str] = {}
    for partition, items in partitions.items():
        for sample_id in items:
            previous = memberships.setdefault(sample_id, partition)
            if previous != partition:
                raise ValueError(
                    f"fold {fold} sample {sample_id} appears in both {previous} and {partition}"
                )
    if available_sample_ids is not None:
        declared = set(memberships)
        missing = sorted(declared - available_sample_ids)
        omitted = sorted(available_sample_ids - declared)
        if missing:
            raise ValueError(f"split manifest references unavailable samples: {missing}")
        if omitted:
            raise ValueError(f"split manifest omits available samples: {omitted}")
    return partitions


def _validate_cv5(folds: dict[int, dict[str, tuple[str, ...]]]) -> None:
    if set(folds) != set(range(5)):
        raise ValueError("cv5 split manifest must contain folds 0 through 4")
    held_out = []
    for fold, partitions in folds.items():
        if len(partitions["train"]) != 8 or partitions["validation"] or len(partitions["test"]) != 2:
            raise ValueError(f"cv5 fold {fold} must contain 8 train, 0 validation, and 2 test samples")
        held_out.extend(partitions["test"])
    if len(held_out) != len(set(held_out)):
        raise ValueError("cv5 test partitions must hold out every sample exactly once")


def _validate_longmemeval_fixed(
    dataset_name: str,
    folds: dict[int, dict[str, tuple[str, ...]]],
) -> None:
    if dataset_name != "longmemeval" or set(folds) != {0}:
        raise ValueError("fixed_80_10_10 is only valid for one LongMemEval fold")
    partitions = folds[0]
    sizes = tuple(len(partitions[name]) for name in PARTITIONS)
    if sizes != (400, 50, 50):
        raise ValueError("LongMemEval fixed split must contain 400 train, 50 validation, and 50 test")


def _validate_longmemeval_cv5(
    dataset_name: str,
    folds: dict[int, dict[str, tuple[str, ...]]],
) -> None:
    if dataset_name != "longmemeval" or set(folds) != set(range(5)):
        raise ValueError("cv5_nested_360_40_100 requires LongMemEval folds 0 through 4")
    held_out = []
    for fold, partitions in folds.items():
        sizes = tuple(len(partitions[name]) for name in PARTITIONS)
        if sizes != (360, 40, 100):
            raise ValueError(
                f"LongMemEval CV fold {fold} must contain 360 train, 40 validation, and 100 test"
            )
        held_out.extend(partitions["test"])
    if len(held_out) != len(set(held_out)):
        raise ValueError("LongMemEval CV test partitions must hold out every sample exactly once")

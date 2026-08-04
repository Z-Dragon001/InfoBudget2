"""Input discovery and replay-ledger validation for router training."""

from __future__ import annotations

import json
from pathlib import Path

from infobudget.rl_router.ledger import read_sqlite_ledger
from infobudget.rl_router.schemas import ReplaySegmentCost, TIERS, TopicSegment


def discover_segment_files(
    root: str | Path,
    dataset_name: str,
    split: str,
    segmentation_method: str,
    sample_ids: list[str] | None = None,
) -> dict[str, Path]:
    samples_dir = Path(root) / "datasets" / "segmented" / dataset_name / split / segmentation_method / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"segmented samples directory is missing: {samples_dir}")
    discovered = {
        path.parent.name: path
        for path in sorted(samples_dir.glob("*/segments.jsonl"))
        if path.is_file()
    }
    if sample_ids:
        missing = sorted(set(sample_ids) - discovered.keys())
        if missing:
            raise FileNotFoundError(f"segmented samples are missing: {missing}")
        discovered = {sample_id: discovered[sample_id] for sample_id in sample_ids}
    if not discovered:
        raise ValueError(f"no segments.jsonl files found under {samples_dir}")
    return discovered


def load_replay_history(
    path: str | Path,
    segments: list[TopicSegment],
    extraction_run_id: str | None = None,
) -> tuple[str, dict[tuple[str, str], ReplaySegmentCost]]:
    source = Path(path)
    if source.suffix.lower() == ".sqlite3" and not source.is_file():
        legacy = source.with_name("segment_costs.jsonl")
        if legacy.is_file():
            source = legacy
    if not source.is_file():
        raise FileNotFoundError(f"candidate segment-cost ledger is missing: {source}")
    if source.suffix.lower() == ".sqlite3":
        values = read_sqlite_ledger(source, "segment_costs")
    else:
        values = []
        with source.open("r", encoding="utf-8") as handle:
            values.extend(json.loads(line) for line in handle if line.strip())
    rows = [value for value in values if value.get("status") in {"ok", "no_fact"}]
    expected = {(segment.segment_id, tier) for segment in segments for tier in TIERS}
    by_run: dict[str, dict[tuple[str, str], ReplaySegmentCost]] = {}
    run_order: list[str] = []
    for row in rows:
        run_id = str(row.get("extraction_run_id") or "")
        tier = str(row.get("tier") or "")
        segment_id = str(row.get("segment_id") or "")
        if not run_id or tier not in TIERS or not segment_id:
            continue
        if run_id not in by_run:
            by_run[run_id] = {}
            run_order.append(run_id)
        key = (segment_id, tier)
        if key in by_run[run_id]:
            raise ValueError(
                f"duplicate replay cost record for {run_id}/{segment_id}/{tier}"
            )
        by_run[run_id][key] = ReplaySegmentCost(
            segment_id=segment_id,
            tier=tier,
            serialized_input_tokens=int(row["serialized_input_tokens"]),
            attributed_output_tokens=int(row["attributed_output_tokens"]),
        )
    if extraction_run_id is not None:
        if extraction_run_id not in by_run:
            raise ValueError(f"extraction run is absent from {source}: {extraction_run_id}")
        selected = extraction_run_id
    else:
        complete = [run_id for run_id in run_order if expected <= by_run[run_id].keys()]
        if not complete:
            raise ValueError(f"no complete L/M/H extraction run exists in {source}")
        selected = complete[-1]
    history = by_run[selected]
    missing = sorted(expected - history.keys())
    if missing:
        preview = ", ".join(f"{segment_id}/{tier}" for segment_id, tier in missing[:5])
        raise ValueError(f"extraction run {selected} is incomplete; missing {preview}")
    return selected, {key: history[key] for key in expected}

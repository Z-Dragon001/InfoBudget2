"""Read-only reconciliation across run manifests, SQLite state/ledgers, and Qdrant."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from infobudget.rl_router.ledger import read_sqlite_ledger
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.schemas import TIERS


class ReconciliationError(RuntimeError):
    pass


def reconcile_extraction_run(
    project_root: str | Path,
    extraction_run_id: str,
    store: FactQdrantStore,
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Verify one sample run without changing its manifest, SQLite files, or Qdrant."""
    root = Path(project_root)
    run_dir = root / "outputs" / "rl_router" / "runs" / extraction_run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"extraction run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if str(manifest.get("extraction_run_id") or "") != extraction_run_id:
        errors.append("manifest extraction_run_id does not match the requested run")
    if manifest.get("qdrant_collection_namespace") != store.namespace:
        errors.append("manifest Qdrant namespace does not match the opened store")

    required_tiers = tuple(manifest.get("required_tiers") or TIERS)
    unknown_tiers = sorted(set(required_tiers) - set(TIERS))
    if unknown_tiers:
        errors.append(f"manifest contains unknown tiers: {unknown_tiers}")
    if manifest.get("status") != "complete":
        errors.append(f"manifest status is not complete: {manifest.get('status')}")
    if set(manifest.get("completed_tiers") or ()) != set(required_tiers):
        errors.append("manifest completed_tiers does not equal required_tiers")

    state_run_status, state_rows = _read_run_state(
        run_dir / "state.sqlite3", extraction_run_id
    )
    if state_run_status != "complete":
        errors.append(f"SQLite run status is not complete: {state_run_status}")
    ledger_path = (
        root
        / "outputs"
        / "rl_router"
        / str(manifest.get("dataset_name") or "")
        / str(manifest.get("split") or "")
        / str(manifest.get("segmentation_method") or "")
        / "samples"
        / str(manifest.get("sample_id") or "")
        / "extraction"
        / "candidate_ledger.sqlite3"
    )
    ledger_rows = read_sqlite_ledger(ledger_path, "segment_costs")
    ledger_rows = [
        row for row in ledger_rows
        if row.get("extraction_run_id") == extraction_run_id
    ]
    summary = manifest.get("extraction_summary") or {}
    planned = manifest.get("planned_extraction") or {}
    summary_statuses = summary.get("batch_status_by_tier") or {}
    summary_fact_counts = summary.get("fact_counts") or {}
    manifest_qdrant_counts = (manifest.get("qdrant_audit") or {}).get("counts_by_tier") or {}
    result_by_tier: dict[str, Any] = {}

    for tier in required_tiers:
        tier_state = [row for row in state_rows if row["tier"] == tier]
        committed_state = [row for row in tier_state if row["status"] == "committed"]
        non_committed = sorted(
            f"{row['batch_id']}={row['status']}"
            for row in tier_state if row["status"] != "committed"
        )
        tier_ledger = [row for row in ledger_rows if row.get("tier") == tier]
        state_batches = {row["batch_id"] for row in committed_state}
        ledger_batches = {str(row.get("batch_id") or "") for row in tier_ledger}
        plan_count = int((planned.get(tier) or {}).get("batch_count", 0))
        manifest_committed = int(
            (summary_statuses.get(tier) or {}).get("committed", 0)
        )
        if non_committed:
            errors.append(f"{tier}: SQLite contains non-committed batches: {non_committed}")
        counts = {
            "manifest_planned_batches": plan_count,
            "manifest_committed_batches": manifest_committed,
            "sqlite_committed_batches": len(committed_state),
            "ledger_distinct_batches": len(ledger_batches),
        }
        if len(set(counts.values())) != 1:
            errors.append(f"{tier}: batch counts disagree: {counts}")
        if state_batches != ledger_batches:
            errors.append(
                f"{tier}: SQLite/segment-ledger batch IDs differ: "
                f"sqlite_only={sorted(state_batches - ledger_batches)}, "
                f"ledger_only={sorted(ledger_batches - state_batches)}"
            )

        state_segments = {
            (row["batch_id"], segment_id)
            for row in committed_state
            for segment_id in row["segment_ids"]
        }
        ledger_segments = {
            (str(row.get("batch_id") or ""), str(row.get("segment_id") or ""))
            for row in tier_ledger
        }
        if state_segments != ledger_segments:
            errors.append(
                f"{tier}: SQLite planned segments differ from segment ledger: "
                f"sqlite_only={sorted(state_segments - ledger_segments)[:10]}, "
                f"ledger_only={sorted(ledger_segments - state_segments)[:10]}"
            )

        points = store.candidate_points(
            tier,
            dataset_name=str(manifest.get("dataset_name") or ""),
            split=str(manifest.get("split") or ""),
            sample_id=str(manifest.get("sample_id") or ""),
            extraction_run_id=extraction_run_id,
            with_vectors=False,
        )
        expected_points = Counter(
            {
                (str(row.get("batch_id") or ""), str(row.get("segment_id") or "")):
                int(row.get("fact_count", 0))
                for row in tier_ledger
            }
        )
        actual_points = Counter(
            (
                str((point.payload or {}).get("batch_id") or ""),
                str((point.payload or {}).get("segment_id") or ""),
            )
            for point in points
        )
        if expected_points != actual_points:
            mismatches = {
                f"{batch_id}/{segment_id}": {
                    "ledger": expected_points[(batch_id, segment_id)],
                    "qdrant": actual_points[(batch_id, segment_id)],
                }
                for batch_id, segment_id in sorted(set(expected_points) | set(actual_points))
                if expected_points[(batch_id, segment_id)]
                != actual_points[(batch_id, segment_id)]
            }
            errors.append(f"{tier}: segment-ledger/Qdrant point counts differ: {mismatches}")
        ledger_fact_count = sum(expected_points.values())
        qdrant_point_count = len(points)
        manifest_fact_count = int(summary_fact_counts.get(tier, 0))
        recorded_qdrant_count = int(manifest_qdrant_counts.get(tier, 0))
        point_counts = {
            "manifest_facts": manifest_fact_count,
            "manifest_qdrant_points": recorded_qdrant_count,
            "ledger_facts": ledger_fact_count,
            "qdrant_points": qdrant_point_count,
        }
        if len(set(point_counts.values())) != 1:
            errors.append(f"{tier}: fact/point counts disagree: {point_counts}")
        result_by_tier[tier] = {
            **counts,
            "sqlite_segment_results": len(state_segments),
            "ledger_segment_results": len(ledger_segments),
            **point_counts,
        }

    result = {
        "schema_version": "extraction_reconciliation_v1",
        "extraction_run_id": extraction_run_id,
        "manifest_path": str(manifest_path),
        "state_path": str(run_dir / "state.sqlite3"),
        "ledger_path": str(ledger_path),
        "qdrant_namespace": store.namespace,
        "passed": not errors,
        "errors": errors,
        "tiers": result_by_tier,
    }
    if errors and raise_on_error:
        raise ReconciliationError(
            f"extraction run {extraction_run_id} failed reconciliation: "
            + " | ".join(errors)
        )
    return result


def _read_run_state(
    path: Path, extraction_run_id: str
) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"extraction SQLite state is missing: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        run = connection.execute(
            "SELECT status FROM runs WHERE run_id = ?", (extraction_run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(f"run is absent from extraction SQLite state: {extraction_run_id}")
        rows = connection.execute(
            "SELECT batch_id, tier, segment_ids_json, status FROM batches "
            "WHERE run_id = ? ORDER BY tier, sequence_index",
            (extraction_run_id,),
        ).fetchall()
        return str(run[0]), [
            {
                "batch_id": str(row[0]),
                "tier": str(row[1]),
                "segment_ids": list(json.loads(str(row[2]))),
                "status": str(row[3]),
            }
            for row in rows
        ]
    finally:
        connection.close()

"""Crash-safe extraction-run state backed by SQLite WAL and an OS file lock."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_BATCH_STATES = {
    "planned",
    "requesting",
    "api_succeeded",
    "fallback_running",
    "parsed",
    "embedded",
}
SUCCESSFUL_BATCH_STATES = {"committed", "recovered_by_fallback"}
FINAL_BATCH_STATES = SUCCESSFUL_BATCH_STATES | {"failed_terminal"}
FALLBACK_BATCH_STATES = {
    "planned",
    "requesting",
    "api_succeeded",
    "validated",
    "committed",
    "failed_retryable",
    "failed_terminal",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunFileLock(AbstractContextManager["RunFileLock"]):
    """Non-blocking process lock whose OS lock is automatically released after a crash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "RunFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"extraction run is already active: {self.path.parent.name}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


class ExtractionRunState:
    """Persist deterministic batch plans and monotonic processing states."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "state.sqlite3"
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                scope_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                segment_ids_json TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, tier, sequence_index),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_batches_run_status
                ON batches(run_id, status);
            CREATE TABLE IF NOT EXISTS fallback_batches (
                fallback_batch_id TEXT PRIMARY KEY,
                parent_batch_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                target_segment_id TEXT NOT NULL,
                child_index INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                context_source_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(parent_batch_id, target_segment_id),
                FOREIGN KEY(parent_batch_id) REFERENCES batches(batch_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fallback_batches_parent
                ON fallback_batches(parent_batch_id, status);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def register_run(self, scope_hash: str, *, resume: bool) -> None:
        row = self.connection.execute(
            "SELECT scope_hash FROM runs WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        now = utc_now()
        if row is None:
            self.connection.execute(
                "INSERT INTO runs(run_id, scope_hash, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (self.run_id, scope_hash, "active", now, now),
            )
        else:
            if not resume:
                raise ValueError(
                    f"extraction run {self.run_id} already exists; use --resume {self.run_id}"
                )
            if str(row["scope_hash"]) != scope_hash:
                raise ValueError("resume scope does not match the original extraction run")
            self.connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                ("active", now, self.run_id),
            )
            self.connection.execute(
                "UPDATE batches SET status = ?, last_error = ?, updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                ("failed_retryable", "recovered stale requesting state", now, self.run_id, "requesting"),
            )
            self.connection.execute(
                "UPDATE fallback_batches SET status = ?, last_error = ?, updated_at = ? "
                "WHERE run_id = ? AND status = ?",
                (
                    "failed_retryable",
                    "recovered stale fallback requesting state",
                    now,
                    self.run_id,
                    "requesting",
                ),
            )
        self.connection.commit()

    def plan_batch(
        self,
        *,
        batch_id: str,
        sample_id: str,
        tier: str,
        sequence_index: int,
        segment_ids: list[str],
        prompt_hash: str,
    ) -> str:
        payload = json.dumps(segment_ids, ensure_ascii=False, separators=(",", ":"))
        row = self.connection.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            now = utc_now()
            self.connection.execute(
                "INSERT INTO batches(batch_id, run_id, sample_id, tier, sequence_index, "
                "segment_ids_json, prompt_hash, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    self.run_id,
                    sample_id,
                    tier,
                    sequence_index,
                    payload,
                    prompt_hash,
                    "planned",
                    now,
                    now,
                ),
            )
            self.connection.commit()
            return "planned"
        expected = (self.run_id, sample_id, tier, sequence_index, payload, prompt_hash)
        actual = tuple(
            row[key]
            for key in (
                "run_id",
                "sample_id",
                "tier",
                "sequence_index",
                "segment_ids_json",
                "prompt_hash",
            )
        )
        if actual != expected:
            raise ValueError(f"batch plan changed while resuming {batch_id}")
        return str(row["status"])

    def status(self, batch_id: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return str(row["status"])

    def mark(self, batch_id: str, status: str, error: str | None = None) -> None:
        allowed = ACTIVE_BATCH_STATES | FINAL_BATCH_STATES | {"failed_retryable"}
        if status not in allowed:
            raise ValueError(f"unknown batch status {status}")
        result = self.connection.execute(
            "UPDATE batches SET status = ?, last_error = ?, updated_at = ? "
            "WHERE batch_id = ? AND run_id = ?",
            (status, error[:2000] if error else None, utc_now(), batch_id, self.run_id),
        )
        if result.rowcount != 1:
            raise KeyError(batch_id)
        self.connection.commit()

    def plan_fallback_batch(
        self,
        *,
        fallback_batch_id: str,
        parent_batch_id: str,
        sample_id: str,
        tier: str,
        target_segment_id: str,
        child_index: int,
        prompt_hash: str,
        context_source_ids: list[int],
    ) -> str:
        context_json = json.dumps(
            context_source_ids, ensure_ascii=False, separators=(",", ":")
        )
        row = self.connection.execute(
            "SELECT * FROM fallback_batches WHERE fallback_batch_id = ?",
            (fallback_batch_id,),
        ).fetchone()
        if row is None:
            now = utc_now()
            self.connection.execute(
                "INSERT INTO fallback_batches("
                "fallback_batch_id, parent_batch_id, run_id, sample_id, tier, "
                "target_segment_id, child_index, prompt_hash, context_source_ids_json, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fallback_batch_id,
                    parent_batch_id,
                    self.run_id,
                    sample_id,
                    tier,
                    target_segment_id,
                    child_index,
                    prompt_hash,
                    context_json,
                    "planned",
                    now,
                    now,
                ),
            )
            self.connection.commit()
            return "planned"
        expected = (
            parent_batch_id,
            self.run_id,
            sample_id,
            tier,
            target_segment_id,
            child_index,
            prompt_hash,
            context_json,
        )
        actual = tuple(
            row[key]
            for key in (
                "parent_batch_id",
                "run_id",
                "sample_id",
                "tier",
                "target_segment_id",
                "child_index",
                "prompt_hash",
                "context_source_ids_json",
            )
        )
        if actual != expected:
            raise ValueError(
                f"fallback batch plan changed while resuming {fallback_batch_id}"
            )
        return str(row["status"])

    def mark_fallback(
        self, fallback_batch_id: str, status: str, error: str | None = None
    ) -> None:
        if status not in FALLBACK_BATCH_STATES:
            raise ValueError(f"unknown fallback batch status {status}")
        result = self.connection.execute(
            "UPDATE fallback_batches SET status = ?, last_error = ?, updated_at = ? "
            "WHERE fallback_batch_id = ? AND run_id = ?",
            (
                status,
                error[:2000] if error else None,
                utc_now(),
                fallback_batch_id,
                self.run_id,
            ),
        )
        if result.rowcount != 1:
            raise KeyError(fallback_batch_id)
        self.connection.commit()

    def fallback_batch_ids(self, parent_batch_ids: list[str]) -> list[str]:
        if not parent_batch_ids:
            return []
        placeholders = ",".join("?" for _ in parent_batch_ids)
        return [
            str(row["fallback_batch_id"])
            for row in self.connection.execute(
                "SELECT fallback_batch_id FROM fallback_batches "
                f"WHERE run_id = ? AND parent_batch_id IN ({placeholders}) "
                "ORDER BY parent_batch_id, child_index",
                [self.run_id, *parent_batch_ids],
            ).fetchall()
        ]

    def terminal_fallback_rows(
        self, tiers: tuple[str, ...] | list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return every singleton child belonging to selected terminal parents."""
        selected = tuple(tiers or ())
        tier_clause = ""
        parameters: list[Any] = [self.run_id, "failed_terminal"]
        if selected:
            tier_clause = f" AND parent.tier IN ({','.join('?' for _ in selected)})"
            parameters.extend(selected)
        rows = self.connection.execute(
            "SELECT parent.batch_id AS parent_batch_id, "
            "parent.sequence_index AS parent_sequence_index, "
            "parent.segment_ids_json AS parent_segment_ids_json, "
            "parent.tier AS parent_tier, parent.last_error AS parent_last_error, "
            "child.fallback_batch_id, child.target_segment_id, child.child_index, "
            "child.context_source_ids_json, child.status AS child_status, "
            "child.last_error AS child_last_error "
            "FROM batches AS parent JOIN fallback_batches AS child "
            "ON child.parent_batch_id = parent.batch_id "
            "WHERE parent.run_id = ? AND parent.status = ?"
            f"{tier_clause} ORDER BY parent.sequence_index, child.child_index",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]

    def terminal_parent_rows(
        self, tiers: tuple[str, ...] | list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return selected terminal parent batches in deterministic order."""
        selected = tuple(tiers or ())
        tier_clause = ""
        parameters: list[Any] = [self.run_id, "failed_terminal"]
        if selected:
            tier_clause = f" AND tier IN ({','.join('?' for _ in selected)})"
            parameters.extend(selected)
        rows = self.connection.execute(
            "SELECT batch_id, sequence_index, segment_ids_json, tier, last_error "
            "FROM batches WHERE run_id = ? AND status = ?"
            f"{tier_clause} ORDER BY tier, sequence_index",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]

    def reset_terminal_parents_for_singletons(
        self, parent_batch_ids: list[str]
    ) -> int:
        """Re-open terminal parents only when they have no singleton children yet."""
        if not parent_batch_ids:
            return 0
        unique_ids = list(dict.fromkeys(parent_batch_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        now = utc_now()
        with self.connection:
            parents = self.connection.execute(
                "SELECT batch_id FROM batches WHERE run_id = ? AND status = ? "
                f"AND batch_id IN ({placeholders})",
                [self.run_id, "failed_terminal", *unique_ids],
            ).fetchall()
            if len(parents) != len(unique_ids):
                raise ValueError(
                    "terminal singleton recovery targets changed before state reset"
                )
            existing_children = self.connection.execute(
                "SELECT DISTINCT parent_batch_id FROM fallback_batches WHERE run_id = ? "
                f"AND parent_batch_id IN ({placeholders})",
                [self.run_id, *unique_ids],
            ).fetchall()
            if existing_children:
                raise ValueError(
                    "terminal singleton recovery requires parents without existing "
                    "singleton children"
                )
            result = self.connection.execute(
                "UPDATE batches SET status = ?, last_error = NULL, updated_at = ? "
                f"WHERE run_id = ? AND status = ? AND batch_id IN ({placeholders})",
                [
                    "failed_retryable",
                    now,
                    self.run_id,
                    "failed_terminal",
                    *unique_ids,
                ],
            )
        return int(result.rowcount)

    def reset_cached_singleton_recovery(self, parent_batch_ids: list[str]) -> int:
        """Re-open prevalidated terminal parents without deleting cached responses."""
        if not parent_batch_ids:
            return 0
        unique_ids = list(dict.fromkeys(parent_batch_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        now = utc_now()
        with self.connection:
            parents = self.connection.execute(
                "SELECT batch_id FROM batches WHERE run_id = ? "
                f"AND status = ? AND batch_id IN ({placeholders})",
                [self.run_id, "failed_terminal", *unique_ids],
            ).fetchall()
            if len(parents) != len(unique_ids):
                raise ValueError(
                    "cached singleton recovery targets changed before state reset"
                )
            self.connection.execute(
                "UPDATE fallback_batches SET status = ?, last_error = NULL, "
                "updated_at = ? WHERE run_id = ? AND status = ? "
                f"AND parent_batch_id IN ({placeholders})",
                [
                    "failed_retryable",
                    now,
                    self.run_id,
                    "failed_terminal",
                    *unique_ids,
                ],
            )
            result = self.connection.execute(
                "UPDATE batches SET status = ?, last_error = NULL, updated_at = ? "
                f"WHERE run_id = ? AND status = ? AND batch_id IN ({placeholders})",
                [
                    "failed_retryable",
                    now,
                    self.run_id,
                    "failed_terminal",
                    *unique_ids,
                ],
            )
        return int(result.rowcount)

    def retry_terminal(self, tiers: tuple[str, ...] | list[str] | None = None) -> int:
        selected = tuple(tiers or ())
        tier_clause = ""
        parameters: list[Any] = ["failed_retryable", utc_now(), self.run_id, "failed_terminal"]
        if selected:
            tier_clause = f" AND tier IN ({','.join('?' for _ in selected)})"
            parameters.extend(selected)
        terminal_parent_ids = [
            str(row["batch_id"])
            for row in self.connection.execute(
                "SELECT batch_id FROM batches WHERE run_id = ? AND status = ?"
                f"{tier_clause}",
                [self.run_id, "failed_terminal", *selected],
            ).fetchall()
        ]
        if terminal_parent_ids:
            placeholders = ",".join("?" for _ in terminal_parent_ids)
            exhausted = self.connection.execute(
                "SELECT DISTINCT parent_batch_id FROM fallback_batches "
                f"WHERE run_id = ? AND parent_batch_id IN ({placeholders})",
                [self.run_id, *terminal_parent_ids],
            ).fetchall()
            if exhausted:
                raise ValueError(
                    "terminal singleton fallback is exhausted; create a new "
                    "extraction run instead of retrying it"
                )
        result = self.connection.execute(
            "UPDATE batches SET status = ?, last_error = NULL, updated_at = ? "
            f"WHERE run_id = ? AND status = ?{tier_clause}",
            parameters,
        )
        if terminal_parent_ids:
            placeholders = ",".join("?" for _ in terminal_parent_ids)
            self.connection.execute(
                "UPDATE fallback_batches SET status = ?, last_error = NULL, updated_at = ? "
                f"WHERE run_id = ? AND parent_batch_id IN ({placeholders})",
                ["failed_retryable", utc_now(), self.run_id, *terminal_parent_ids],
            )
        self.connection.commit()
        return int(result.rowcount)

    def batch_ids(
        self, status: str, tiers: tuple[str, ...] | list[str] | None = None
    ) -> list[str]:
        selected = tuple(tiers or ())
        tier_clause = ""
        parameters: list[Any] = [self.run_id, status]
        if selected:
            tier_clause = f" AND tier IN ({','.join('?' for _ in selected)})"
            parameters.extend(selected)
        return [
            str(row["batch_id"])
            for row in self.connection.execute(
                "SELECT batch_id FROM batches WHERE run_id = ? AND status = ?"
                f"{tier_clause} ORDER BY tier, sequence_index",
                parameters,
            ).fetchall()
        ]

    def finish_run(self, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, utc_now(), self.run_id),
        )
        self.connection.commit()

    def summary(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT tier, status, COUNT(*) AS count FROM batches WHERE run_id = ? "
            "GROUP BY tier, status",
            (self.run_id,),
        ).fetchall()
        by_status: dict[str, int] = {}
        by_tier: dict[str, dict[str, int]] = {}
        for row in rows:
            status, count, tier = str(row["status"]), int(row["count"]), str(row["tier"])
            by_status[status] = by_status.get(status, 0) + count
            by_tier.setdefault(tier, {})[status] = count
        return {
            "batch_count": sum(by_status.values()),
            "by_status": by_status,
            "by_tier": by_tier,
        }

"""Append-only, deduplicated JSONL ledgers."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlLedger:
    """Legacy single-process ledger kept only for reading old experiment artifacts."""

    def __init__(self, path: str | Path, key_fields: tuple[str, ...] = ()): 
        self.path = Path(path)
        self.key_fields = key_fields
        self._lock = Lock()
        self._keys: set[tuple[Any, ...]] = set()
        if self.path.is_file() and key_fields:
            for row in self.read_all():
                self._keys.add(tuple(row.get(field) for field in key_fields))

    def append(self, row: dict[str, Any]) -> bool:
        key = tuple(row.get(field) for field in self.key_fields)
        with self._lock:
            if self.key_fields and key in self._keys:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.key_fields:
                self._keys.add(key)
        return True

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class SqliteLedger:
    """Cross-process append-only ledger with database-enforced idempotency."""

    def __init__(
        self,
        path: str | Path,
        table: str,
        key_fields: tuple[str, ...] = (),
        *,
        legacy_jsonl_path: str | Path | None = None,
        timeout_seconds: float = 30.0,
    ):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"invalid SQLite ledger table name: {table!r}")
        self.path = Path(path)
        self.table = table
        self.key_fields = key_fields
        self.timeout_seconds = float(timeout_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        initialize_wal = not self.path.exists()
        for attempt in range(20):
            try:
                with self._connect() as connection:
                    if initialize_wal:
                        connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute(
                        f'CREATE TABLE IF NOT EXISTS "{self.table}" ('
                        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "dedupe_key TEXT NOT NULL UNIQUE, "
                        "row_json TEXT NOT NULL)"
                    )
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 19:
                    raise
                initialize_wal = False
                time.sleep(0.01 * (attempt + 1))
        if legacy_jsonl_path is not None:
            self._import_legacy_jsonl(Path(legacy_jsonl_path))

    def append(self, row: dict[str, Any]) -> bool:
        key = self._dedupe_key(row)
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            cursor = connection.execute(
                f'INSERT OR IGNORE INTO "{self.table}"(dedupe_key, row_json) VALUES (?, ?)',
                (key, payload),
            )
            return cursor.rowcount == 1

    def upsert(self, row: dict[str, Any]) -> None:
        """Atomically replace the current projection for one logical ledger key."""
        if not self.key_fields:
            raise ValueError("SQLite ledger upsert requires key_fields")
        key = self._dedupe_key(row)
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                f'INSERT INTO "{self.table}"(dedupe_key, row_json) VALUES (?, ?) '
                "ON CONFLICT(dedupe_key) DO UPDATE SET row_json = excluded.row_json",
                (key, payload),
            )

    def read_all(self) -> list[dict[str, Any]]:
        return read_sqlite_ledger(self.path, self.table)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _dedupe_key(self, row: dict[str, Any]) -> str:
        if not self.key_fields:
            return str(uuid.uuid4())
        missing = [field for field in self.key_fields if field not in row]
        if missing:
            raise ValueError(
                f"SQLite ledger row is missing key fields: {', '.join(missing)}"
            )
        return json.dumps(
            [row[field] for field in self.key_fields],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _import_legacy_jsonl(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.append(json.loads(line))


def read_sqlite_ledger(path: str | Path, table: str) -> list[dict[str, Any]]:
    """Read an existing ledger without creating or modifying the database."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"invalid SQLite ledger table name: {table!r}")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SQLite ledger is missing: {source}")
    connection = sqlite3.connect(
        source.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=30.0,
    )
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"SQLite ledger table is missing: {table}")
        rows = connection.execute(
            f'SELECT row_json FROM "{table}" ORDER BY sequence'
        ).fetchall()
        return [json.loads(str(row[0])) for row in rows]
    finally:
        connection.close()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Replace one JSON document atomically without leaving a partial manifest/archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target

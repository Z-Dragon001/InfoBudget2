"""Strict JSON/JSONL helpers for quality-router artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from infobudget.quality_router.schemas import ModelCapabilityProfile


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    paths = [source] if source.is_file() else sorted(source.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found: {source}")
    for item in paths:
        with item.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object: {item}:{line_number}")
                yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def load_capability_profiles(path: str | Path) -> dict[str, ModelCapabilityProfile]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "memoryprint_v1":
        raise ValueError("model capability schema_version must be memoryprint_v1")
    profiles = [ModelCapabilityProfile.from_dict(item) for item in payload.get("profiles", [])]
    if not profiles:
        raise ValueError("model capability file contains no profiles")
    by_model: dict[str, ModelCapabilityProfile] = {}
    profile_ids: set[str] = set()
    for profile in profiles:
        if profile.model_id in by_model:
            raise ValueError(f"duplicate capability model_id: {profile.model_id}")
        if profile.profile_id in profile_ids:
            raise ValueError(f"duplicate capability profile_id: {profile.profile_id}")
        by_model[profile.model_id] = profile
        profile_ids.add(profile.profile_id)
    return by_model


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

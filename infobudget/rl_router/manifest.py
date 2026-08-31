"""Reproducible experiment manifest generation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.config import RLConfigBundle, scan_config_secrets
from infobudget.rl_router.embedding import directory_hash
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.schemas import QDRANT_FACT_SCHEMA_VERSION


def create_experiment_manifest(
    bundle: RLConfigBundle,
    output_path: str | Path,
    *,
    precomputed_embedding_hash: str | None = None,
    **scope,
) -> dict:
    secrets = scan_config_secrets(bundle.config_dir)
    if secrets:
        raise ValueError(f"refusing to write manifest while config secrets exist: {secrets}")
    embedding = bundle.embeddings["memory"]
    model_path = _resolve(bundle.project.root_dir, embedding["local_path"])
    manifest = {
        "experiment_id": scope.pop("experiment_id", datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")),
        **scope,
        "project_name": bundle.project.config.project.name,
        "model_family": bundle.rl["model_family"],
        "random_seed": bundle.rl["seed"],
        "models": {name: _safe_model(spec) for name, spec in bundle.project.models.items()},
        "price_snapshot": {name: vars_safe(price) for name, price in bundle.project.prices.items()},
        "embedding_model": embedding["model_name"],
        "embedding_model_hash": precomputed_embedding_hash or directory_hash(model_path),
        "prompt_hashes": {
            role: hashlib.sha256(bundle.prompt_path(role).read_bytes()).hexdigest()
            for role in bundle.rl["prompts"]
        },
        "buffer_config": bundle.rl["extraction"],
        "router_config": bundle.rl["router"],
        "qdrant_point_schema_version": QDRANT_FACT_SCHEMA_VERSION,
        "qdrant_storage": safe_qdrant_storage_config(bundle.rl["storage"]),
        "git_commit": _git_commit(bundle.project.root_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(output_path)
    atomic_write_json(path, manifest)
    return manifest


def memory_embedding_hash(bundle: RLConfigBundle) -> str:
    embedding = bundle.embeddings["memory"]
    return directory_hash(_resolve(bundle.project.root_dir, embedding["local_path"]))


def resolve_collection_namespace(
    storage: dict,
    *,
    project_name: str,
    dataset: str,
    split: str,
    segmentation_version: str,
    embedding_hash: str,
    model_family: str,
) -> str:
    if not embedding_hash:
        raise ValueError("embedding_hash is required for the Qdrant namespace")
    normalized_family = str(model_family).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized_family):
        raise ValueError("model_family is invalid for the Qdrant namespace")
    normalized_project = re.sub(
        r"[^a-z0-9_-]+", "-", str(project_name).strip().lower()
    ).strip("-_")
    if not normalized_project:
        raise ValueError("project_name is invalid for the Qdrant namespace")
    return str(storage["collection_namespace"]).format(
        project_name=normalized_project,
        model_family=normalized_family,
        dataset=dataset,
        split=split,
        segmentation_version=segmentation_version,
        embedding_hash=embedding_hash[:12],
    )


def update_experiment_manifest(output_path: str | Path, **updates) -> dict:
    """Atomically update runtime/audit fields while preserving the immutable run scope."""
    path = Path(output_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, manifest)
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vars_safe(value) -> dict:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _safe_model(spec) -> dict:
    value = vars_safe(spec)
    value.pop("api_key", None)
    return value


def safe_qdrant_storage_config(storage: dict) -> dict:
    """Persist connection identity, never a resolved API key."""
    allowed = {
        "mode",
        "url",
        "grpc_port",
        "prefer_grpc",
        "timeout_seconds",
        "api_key_env",
        "local_path",
        "collection_namespace",
        "vector_size",
        "distance",
        "require_sample_filter",
        "require_assembly_filter_for_s",
    }
    return {key: storage[key] for key in allowed if key in storage}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

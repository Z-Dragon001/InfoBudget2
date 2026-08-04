"""Read-only manifest/SQLite/Qdrant reconciliation for one extraction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.reconciliation import reconcile_extraction_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extraction_run_id")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    bundle = load_rl_bundle(args.config_dir)
    manifest_path = (
        bundle.project.root_dir
        / "outputs"
        / "rl_router"
        / "runs"
        / args.extraction_run_id
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    storage = dict(manifest.get("qdrant_storage") or bundle.rl["storage"])
    FactQdrantStore.probe_storage_config(storage)
    store = FactQdrantStore.from_storage_config(
        storage,
        project_root=bundle.project.root_dir,
        namespace=str(manifest["qdrant_collection_namespace"]),
        read_only=True,
    )
    try:
        result = reconcile_extraction_run(
            bundle.project.root_dir,
            args.extraction_run_id,
            store,
        )
    finally:
        store.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

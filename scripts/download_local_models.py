"""Explicitly download configured local models outside training/evaluation flows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a configured embedding model from Hugging Face.")
    parser.add_argument("--role", choices=["router", "memory"], default="memory")
    parser.add_argument("--config", type=Path, default=Path("configs/embeddings.yaml"))
    parser.add_argument("--revision", default=None, help="Optional immutable commit hash or tag.")
    parser.add_argument("--force", action="store_true", help="Allow downloading into an existing directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    spec = raw.get("embeddings", {}).get(args.role)
    if not isinstance(spec, dict):
        raise ValueError(f"embedding role {args.role!r} is missing from {config_path}")
    repo_id = str(spec["model_name"])
    configured = Path(str(spec["local_path"]))
    target = configured if configured.is_absolute() else (project_root / configured).resolve()
    if target.exists() and any(target.iterdir()) and not args.force:
        raise FileExistsError(f"target is not empty; inspect it or rerun with --force: {target}")
    target.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    requested_revision = args.revision or spec.get("revision")
    resolved_revision = api.model_info(repo_id=repo_id, revision=requested_revision).sha
    files = api.list_repo_files(repo_id=repo_id, revision=resolved_revision)
    weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    allow_patterns = [
        weight,
        "config.json",
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "1_Pooling/*",
        "README.md",
    ]
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        local_dir=target,
        allow_patterns=allow_patterns,
        max_workers=1,
    )
    manifest = {
        "model_name": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "resolved_path": str(Path(resolved).resolve()),
        "dimension": int(spec["dimension"]),
        "max_length": int(spec["max_length"]),
        "normalize": bool(spec["normalize"]),
        "files": _file_hashes(target),
    }
    (target / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"model_name": repo_id, "target": str(target), "weight": weight}, ensure_ascii=False))


def _file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "download_manifest.json" and ".cache" not in item.parts
    ):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


if __name__ == "__main__":
    main()

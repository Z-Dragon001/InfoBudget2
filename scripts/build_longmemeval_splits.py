"""Generate leakage-safe fixed and five-fold LongMemEval split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.datasets.split_builder import build_longmemeval_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="nsp_text_tiling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/splits/longmemeval"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fixed, cv5 = build_longmemeval_manifests(
        root,
        segmentation_method=args.method,
        seed=args.seed,
    )
    outputs = {
        args.output_dir / f"fixed_80_10_10_seed{args.seed}_{args.method}.json": fixed,
        args.output_dir / f"cv5_360_40_100_seed{args.seed}_{args.method}.json": cv5,
    }
    for path, payload in outputs.items():
        if path.exists() and not args.force:
            raise FileExistsError(f"split manifest already exists; pass --force to replace it: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.resolve())


if __name__ == "__main__":
    main()

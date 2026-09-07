"""Audit frozen Reference Facts and create a deterministic human-review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_fact_pipeline.audit import audit_reference_facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-archive-dir", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--random-sample-size", type=int, default=120)
    parser.add_argument("--max-facts-per-segment", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = audit_reference_facts(
        references_path=args.references,
        segments_path=args.segments,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        raw_archive_dir=args.raw_archive_dir,
        ledger_path=args.ledger,
        embedding_model_path=args.embedding_model_path,
        embedding_model_name=args.embedding_model_name,
        embedding_dimension=args.embedding_dimension,
        similarity_threshold=args.similarity_threshold,
        random_sample_size=args.random_sample_size,
        max_facts_per_segment=args.max_facts_per_segment,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


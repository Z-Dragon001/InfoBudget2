"""Persist deterministic, sample-isolated segmentation artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from infobudget.config import ProjectBundle
from infobudget.datasets.loader import DatasetLoader
from infobudget.segmentation.factory import build_segmenter


@dataclass(slots=True)
class SegmentationRun:
    bundle: ProjectBundle

    def run(self, dataset_name: str, split: str) -> dict:
        method = self.bundle.config.segmentation.method
        version = f"{method}_v1"
        loader = DatasetLoader(self.bundle.config.dataset, self.bundle.root_dir)
        examples = loader.load(dataset_name, split)
        processed_dir = loader.split_dir(dataset_name, split)
        processed_manifest = processed_dir / "manifest.json"
        output = self.bundle.root_dir / "datasets" / "segmented" / dataset_name / split / method
        output.mkdir(parents=True, exist_ok=True)
        segmenter = build_segmenter(self.bundle.config.segmentation, self.bundle.root_dir)
        total_segments = 0
        total_turns = 0
        for example in examples:
            segments, trace = segmenter.segment_with_trace(example.dialogue)
            sample_dir = output / "samples" / example.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            turn_by_id = {turn.turn_id: turn for turn in example.dialogue}
            session_by_turn = {
                turn.turn_id: session.session_id for session in example.sessions for turn in session.turns
            }
            rows = []
            for index, segment in enumerate(segments, start=1):
                turns = [turn_by_id[item] for item in segment.turn_ids]
                segment_id = f"{example.sample_id}:{method}:seg_{index:06d}"
                segment.segment_id = segment_id
                content_hash = hashlib.sha256(segment.text.encode("utf-8")).hexdigest()
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "split": split,
                        "sample_id": example.sample_id,
                        "session_id": session_by_turn[segment.start_turn],
                        "segmentation_method": method,
                        "segmentation_version": version,
                        "preprocessing_version": self.bundle.config.dataset.schema_version,
                        "segment_id": segment_id,
                        "segment_index": index,
                        "segment_order": index,
                        "start_turn": segment.start_turn,
                        "end_turn": segment.end_turn,
                        "turn_ids": segment.turn_ids,
                        "start_timestamp": turns[0].timestamp,
                        "end_timestamp": turns[-1].timestamp,
                        "text": segment.text,
                        "token_count": segment.token_count,
                        "mean_coherence_score": segment.mean_adjacent_similarity,
                        "boundary_reason": segment.boundary_reason,
                        "source_turn_count": len(turns),
                        "source_content_hash": content_hash,
                        "model_path": self.bundle.config.segmentation.bert_model_dir,
                        "checkpoint_path": self.bundle.config.segmentation.bert_mlp_checkpoint if method == "bert_mlp_text_tiling" else None,
                        "bert_max_length": self.bundle.config.segmentation.bert_max_length,
                        "adaptive_alpha": self.bundle.config.segmentation.adaptive_alpha,
                    }
                )
            self._write_jsonl(sample_dir / "segments.jsonl", rows)
            trace.update(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "sample_id": example.sample_id,
                    "segmentation_method": method,
                    "segmentation_version": version,
                    "preprocessing_manifest_hash": self._hash_file(processed_manifest),
                    "num_input_turns": len(example.dialogue),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            (sample_dir / "segmentation_trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_segments += len(rows)
            total_turns += len(example.dialogue)
        manifest = {
            "dataset_name": dataset_name,
            "split": split,
            "segmentation_method": method,
            "segmentation_version": version,
            "processed_data_dir": str(processed_dir),
            "processed_manifest_hash": self._hash_file(processed_manifest),
            "parameters": asdict(self.bundle.config.segmentation),
            "num_samples": len(examples),
            "num_turns": total_turns,
            "num_segments": total_segments,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _hash_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

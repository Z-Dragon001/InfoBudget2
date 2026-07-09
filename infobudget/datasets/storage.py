"""功能：定义 processed 数据集工件的存储与加载方式。
输入：统一样本列表。
输出：sample / question / session 三级 JSONL 与 manifest。
依赖：json、pathlib、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from infobudget.config import DatasetConfig
from infobudget.schemas import DatasetDialogueExample


@dataclass(slots=True)
class DatasetArtifactStore:
    """processed 数据集工件存储器。"""

    cfg: DatasetConfig
    root_dir: Path

    def split_dir(self, dataset_name: str, split: str) -> Path:
        """返回 split 目录。"""
        return (self.root_dir / self.cfg.processed_dir / dataset_name / split).resolve()

    def save_split(
        self,
        dataset_name: str,
        split: str,
        examples: list[DatasetDialogueExample],
        *,
        raw_files: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """保存一个 split 的所有工件。"""
        split_dir = self.split_dir(dataset_name, split)
        split_dir.mkdir(parents=True, exist_ok=True)
        samples_path = split_dir / "samples.jsonl"
        questions_path = split_dir / "questions.jsonl"
        sessions_path = split_dir / "sessions.jsonl"
        manifest_path = split_dir / "manifest.json"

        self._write_jsonl(samples_path, [example.to_dict() for example in examples])
        if self.cfg.store_flat_questions:
            self._write_jsonl(questions_path, list(self._flatten_questions(examples)))
        if self.cfg.store_flat_sessions:
            self._write_jsonl(sessions_path, list(self._flatten_sessions(examples)))

        manifest = {
            "dataset_name": dataset_name,
            "split": split,
            "layout_version": 2,
            "num_samples": len(examples),
            "num_questions": sum(len(example.qa_pairs) for example in examples),
            "num_sessions": sum(len(example.sessions) for example in examples),
            "raw_files": raw_files or [],
            "files": {
                "samples": str(samples_path),
                "questions": str(questions_path),
                "sessions": str(sessions_path),
            },
            "metadata": metadata or {},
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return manifest

    def save_split_stream(
        self,
        dataset_name: str,
        split: str,
        examples,
        *,
        raw_files: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """流式保存一个 split 的所有工件。"""
        split_dir = self.split_dir(dataset_name, split)
        split_dir.mkdir(parents=True, exist_ok=True)
        samples_path = split_dir / "samples.jsonl"
        questions_path = split_dir / "questions.jsonl"
        sessions_path = split_dir / "sessions.jsonl"
        manifest_path = split_dir / "manifest.json"

        num_samples = 0
        num_questions = 0
        num_sessions = 0
        with (
            samples_path.open("w", encoding="utf-8") as sample_handle,
            questions_path.open("w", encoding="utf-8") as question_handle,
            sessions_path.open("w", encoding="utf-8") as session_handle,
        ):
            for example in examples:
                sample_handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                num_samples += 1
                if self.cfg.store_flat_questions:
                    for question in example.qa_pairs:
                        question_handle.write(
                            json.dumps(
                                {
                                    "dataset_name": example.dataset_name,
                                    "split": example.split,
                                    "sample_id": example.sample_id,
                                    **asdict(question),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        num_questions += 1
                else:
                    num_questions += len(example.qa_pairs)
                if self.cfg.store_flat_sessions:
                    for session in example.sessions:
                        session_handle.write(
                            json.dumps(
                                {
                                    "dataset_name": example.dataset_name,
                                    "split": example.split,
                                    "sample_id": example.sample_id,
                                    **asdict(session),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        num_sessions += 1
                else:
                    num_sessions += len(example.sessions)

        manifest = {
            "dataset_name": dataset_name,
            "split": split,
            "layout_version": 2,
            "num_samples": num_samples,
            "num_questions": num_questions,
            "num_sessions": num_sessions,
            "raw_files": raw_files or [],
            "files": {
                "samples": str(samples_path),
                "questions": str(questions_path),
                "sessions": str(sessions_path),
            },
            "metadata": metadata or {},
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return manifest

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _flatten_questions(examples: list[DatasetDialogueExample]):
        for example in examples:
            for question in example.qa_pairs:
                yield {
                    "dataset_name": example.dataset_name,
                    "split": example.split,
                    "sample_id": example.sample_id,
                    **asdict(question),
                }

    @staticmethod
    def _flatten_sessions(examples: list[DatasetDialogueExample]):
        for example in examples:
            for session in example.sessions:
                yield {
                    "dataset_name": example.dataset_name,
                    "split": example.split,
                    "sample_id": example.sample_id,
                    **asdict(session),
                }

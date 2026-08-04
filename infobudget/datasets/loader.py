"""功能：加载统一预处理后的数据集工件。
输入：数据集名、split 或工件路径。
输出：DatasetDialogueExample / questions / sessions。
依赖：json、pathlib、config、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from infobudget.config import DatasetConfig
from infobudget.schemas import DatasetDialogueExample, DatasetQAPair, DatasetSession, Turn


@dataclass(slots=True)
class DatasetLoader:
    """统一数据集 loader。"""

    cfg: DatasetConfig
    root_dir: Path

    def split_dir(self, dataset_name: str, split: str) -> Path:
        """返回 processed split 目录。"""
        return (self.root_dir / self.cfg.processed_dir / dataset_name / split).resolve()

    def manifest_path(self, dataset_name: str, split: str) -> Path:
        """返回 manifest 路径。"""
        return self.split_dir(dataset_name, split) / "manifest.json"

    def load_manifest(self, dataset_name: str, split: str) -> dict:
        """加载 split manifest。"""
        path = self.manifest_path(dataset_name, split)
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_available_splits(self, dataset_name: str) -> list[str]:
        """列出当前数据集已有的 split。"""
        dataset_dir = (self.root_dir / self.cfg.processed_dir / dataset_name).resolve()
        if not dataset_dir.exists():
            return []
        return sorted([item.name for item in dataset_dir.iterdir() if item.is_dir()])

    def load(
        self,
        dataset_name: str,
        split: str,
        sample_ids: set[str] | None = None,
    ) -> list[DatasetDialogueExample]:
        """按数据集与 split 加载样本。"""
        manifest = self.load_manifest(dataset_name, split)
        return self.load_samples_from_path(Path(manifest["files"]["samples"]), sample_ids)

    def iter_samples(self, dataset_name: str, split: str, sample_ids: set[str] | None = None):
        """按数据集与 split 流式加载样本。"""
        manifest = self.load_manifest(dataset_name, split)
        yield from self.iter_samples_from_path(Path(manifest["files"]["samples"]), sample_ids)

    def load_questions(self, dataset_name: str, split: str) -> list[dict]:
        """加载扁平问题索引。"""
        manifest = self.load_manifest(dataset_name, split)
        return self._read_jsonl(Path(manifest["files"]["questions"]))

    def load_sessions(self, dataset_name: str, split: str) -> list[dict]:
        """加载扁平 session 索引。"""
        manifest = self.load_manifest(dataset_name, split)
        return self._read_jsonl(Path(manifest["files"]["sessions"]))

    @staticmethod
    def load_samples_from_path(
        path: Path,
        sample_ids: set[str] | None = None,
    ) -> list[DatasetDialogueExample]:
        """按路径加载 sample 文件。"""
        return list(DatasetLoader.iter_samples_from_path(path, sample_ids))

    @staticmethod
    def iter_samples_from_path(path: Path, sample_ids: set[str] | None = None):
        """按路径流式加载 sample 文件。"""
        if not path.exists():
            raise FileNotFoundError(f"Processed sample file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if sample_ids is not None and raw["sample_id"] not in sample_ids:
                    continue
                yield DatasetDialogueExample(
                    sample_id=raw["sample_id"],
                    dataset_name=raw["dataset_name"],
                    split=raw["split"],
                    sessions=[
                        DatasetSession(
                            session_id=session["session_id"],
                            timestamp=session.get("timestamp"),
                            raw_timestamp=session.get("raw_timestamp"),
                            turns=[Turn(**turn) for turn in session["turns"]],
                            metadata=session.get("metadata", {}),
                        )
                        for session in raw.get("sessions", [])
                    ],
                    dialogue=[Turn(**turn) for turn in raw["dialogue"]],
                    qa_pairs=[DatasetQAPair(**qa) for qa in raw["qa_pairs"]],
                    metadata=raw.get("metadata", {}),
                )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

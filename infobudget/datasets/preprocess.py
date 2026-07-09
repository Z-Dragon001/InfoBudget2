"""功能：批量预处理 LoCoMo 与 LongMemEval 数据集。
输入：dataset 配置、数据集名与 raw 文件。
输出：processed 工件目录与 manifest。
依赖：pathlib、config、registry、storage。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from infobudget.config import DatasetConfig
from infobudget.datasets.base import iter_raw_records
from infobudget.datasets.registry import DatasetRegistry
from infobudget.datasets.storage import DatasetArtifactStore
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DatasetPreprocessManager:
    """数据集预处理调度器。"""

    cfg: DatasetConfig
    root_dir: Path
    store: DatasetArtifactStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = DatasetArtifactStore(self.cfg, self.root_dir)

    def preprocess_file(self, dataset_name: str, raw_path: Path, split: str | None = None) -> dict:
        """处理单个原始文件。"""
        resolved_split = split or self._infer_split_from_name(raw_path.name)
        preprocessor = DatasetRegistry.create(dataset_name)
        manifest = self.store.save_split_stream(
            dataset_name,
            resolved_split,
            preprocessor.iter_examples(iter_raw_records(raw_path), split=resolved_split),
            raw_files=[str(raw_path.resolve())],
            metadata={"source_file": raw_path.name},
        )
        logger.info("Preprocessed %s/%s from %s", dataset_name, resolved_split, raw_path.name)
        return manifest

    def preprocess_dataset(self, dataset_name: str, split: str | None = None) -> dict[str, int]:
        """预处理某个数据集目录下的所有原始文件。"""
        dataset_dir = (self.root_dir / self.cfg.raw_dir / dataset_name).resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {dataset_dir}")
        split_to_files: dict[str, list[Path]] = {}
        for raw_path in sorted(dataset_dir.iterdir()):
            if not raw_path.is_file() or raw_path.name.startswith("."):
                continue
            resolved_split = split or self._infer_split_from_name(raw_path.name)
            split_to_files.setdefault(resolved_split, []).append(raw_path)

        summary: dict[str, int] = {}
        preprocessor = DatasetRegistry.create(dataset_name)
        for resolved_split, raw_files in split_to_files.items():
            manifest = self.store.save_split_stream(
                dataset_name,
                resolved_split,
                self._iter_examples_from_files(preprocessor, raw_files, resolved_split),
                raw_files=[str(path.resolve()) for path in raw_files],
                metadata={"source_files": [path.name for path in raw_files]},
            )
            summary[resolved_split] = int(manifest["num_samples"])
            logger.info(
                "Preprocessed %s/%s from %s file(s)",
                dataset_name,
                resolved_split,
                len(raw_files),
            )
        return summary

    def preprocess_all(self, datasets: list[str] | None = None) -> dict[str, dict[str, int]]:
        """批量预处理全部支持数据集。"""
        dataset_names = datasets or self.cfg.supported_datasets
        summary: dict[str, dict[str, int]] = {}
        for dataset_name in dataset_names:
            try:
                summary[dataset_name] = self.preprocess_dataset(dataset_name)
            except FileNotFoundError:
                logger.warning("Skip missing raw dataset dir: %s", dataset_name)
                summary[dataset_name] = {}
        return summary

    def _infer_split_from_name(self, file_name: str) -> str:
        lowered = file_name.lower()
        tokens: list[str] = []
        current = ""
        for ch in lowered:
            if ch.isalnum():
                current += ch
            elif current:
                tokens.append(current)
                current = ""
        if current:
            tokens.append(current)
        for candidate in ["train", "dev", "val", "valid", "validation", "test"]:
            if candidate in tokens:
                if candidate in {"valid", "validation"}:
                    return "val"
                return candidate
        return self.cfg.fallback_split_name

    @staticmethod
    def _iter_examples_from_files(preprocessor, raw_files: list[Path], split: str):
        for raw_path in raw_files:
            yield from preprocessor.iter_examples(iter_raw_records(raw_path), split=split)

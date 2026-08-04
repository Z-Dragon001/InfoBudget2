"""Construct one of the two formal local-BERT segmentation methods."""

from __future__ import annotations

from pathlib import Path

from infobudget.config import SegmentationConfig
from infobudget.segmentation.base import BaseSegmenter
from infobudget.segmentation.bert_mlp_text_tiling import BertMLPTextTilingSegmenter
from infobudget.segmentation.nsp_text_tiling import NSPTextTilingSegmenter


def build_segmenter(cfg: SegmentationConfig, root_dir: str | Path) -> BaseSegmenter:
    method = cfg.method.strip().lower().replace("-", "_")
    root = Path(root_dir).resolve()
    model_dir = _resolve(root, cfg.bert_model_dir)
    checkpoint = _resolve(root, cfg.bert_mlp_checkpoint)
    if method == "nsp_text_tiling":
        return NSPTextTilingSegmenter(cfg, model_dir)
    if method == "bert_mlp_text_tiling":
        return BertMLPTextTilingSegmenter(cfg, model_dir, checkpoint)
    raise ValueError(
        f"Unknown segmentation.method={cfg.method!r}; expected nsp_text_tiling or bert_mlp_text_tiling"
    )


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()

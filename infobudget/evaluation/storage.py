"""功能：存储评估结果集、检索轨迹和运行清单。
输入：metrics、predictions、retrieval traces。
输出：结构化评估结果目录。
依赖：json、pathlib、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from infobudget.evaluation.metrics import EvaluationMetrics
from infobudget.schemas import RetrievalTrace


@dataclass(slots=True)
class EvaluationArtifactStore:
    """评估工件存储器。"""

    root_dir: Path

    def save(
        self,
        *,
        dataset_name: str,
        split: str,
        metrics: EvaluationMetrics,
        predictions: list[dict],
        retrieval_traces: list[RetrievalTrace],
        metadata: dict,
    ) -> Path:
        """写出本次评估的全部工件。"""
        output_dir = (self.root_dir / "outputs" / "evaluation" / dataset_name / split).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics.to_dict(), handle, ensure_ascii=False, indent=2)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / "retrieval_traces.jsonl").open("w", encoding="utf-8") as handle:
            for trace in retrieval_traces:
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return output_dir

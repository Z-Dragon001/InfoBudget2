"""功能：记录模型调用成本与延迟。
输入：模型调用元数据。
输出：CostLogEntry 列表与 JSONL 文件。
依赖：pathlib、schemas、runtime。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from infobudget.runtime.model_registry import PriceRegistry
from infobudget.schemas import CostLogEntry, ModelSpec, Tier
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CostLogger:
    """统一成本日志记录器。"""

    price_registry: PriceRegistry
    output_path: Path
    logs: list[CostLogEntry] = field(default_factory=list)

    def log_extraction(
        self,
        *,
        segment_id: str,
        tier: Tier,
        model_spec: ModelSpec,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        extraction_mode: str = "joint",
    ) -> CostLogEntry:
        """记录一次提取调用。"""
        cost_usd = self.price_registry.estimate_cost(model_spec.model_name, input_tokens, output_tokens)
        log_entry = CostLogEntry(
            call_id=f"call_{len(self.logs)+1:06d}",
            segment_id=segment_id,
            tier=tier,
            model_name=model_spec.model_name,
            backend=model_spec.deploy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 8),
            latency_ms=latency_ms,
            extraction_mode=extraction_mode,
        )
        self.logs.append(log_entry)
        logger.info("Logged model call %s for segment %s", log_entry.call_id, segment_id)
        return log_entry

    def save(self) -> None:
        """持久化日志到 JSONL。"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for item in self.logs:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

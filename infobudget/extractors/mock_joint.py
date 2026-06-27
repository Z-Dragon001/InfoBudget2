"""功能：实现用于测试和本地联调的 MockJointExtractor。
输入：Segment、tier、ScoreResult。
输出：结构化 MemoryEntry，并记录成本日志。
依赖：配置、成本日志、注册表、text。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.cost.logger import CostLogger
from infobudget.extractors.base import JointMemoryExtractor
from infobudget.runtime.model_registry import ModelRegistry
from infobudget.schemas import (
    Constraint,
    Episode,
    EpisodicMemory,
    MemoryEntry,
    Preference,
    ScoreResult,
    Segment,
    SemanticEntity,
    SemanticFact,
    SemanticMemory,
    Tier,
)
from infobudget.utils.text import (
    clamp01,
    count_tokens,
    detect_topic,
    extract_constraints,
    extract_entities,
    extract_event_clauses,
    extract_preferences,
    summarize_text,
)


@dataclass(slots=True)
class MockJointExtractor(JointMemoryExtractor):
    """不调用真实 LLM 的联合提取器。"""

    model_registry: ModelRegistry
    cost_logger: CostLogger
    prompt_template: str

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> MemoryEntry:
        model_spec = self.model_registry.get(tier)
        entities = [
            SemanticEntity(name=name, type="ENTITY", aliases=[])
            for name in extract_entities(segment.text)
        ]
        facts = self._build_facts(segment.text, entities)
        preferences = [
            Preference(owner=owner, item="preference", value=value, confidence=0.75)
            for owner, value in extract_preferences(segment.text)
        ]
        constraints = [
            Constraint(owner="user", rule=rule, confidence=0.82)
            for rule in extract_constraints(segment.text)
        ]
        episodes = [
            Episode(
                subject="user",
                verb="discuss",
                object=event,
                time="",
                location="",
                sequence=index,
                cause="",
                result="",
                confidence=0.72,
            )
            for index, event in enumerate(extract_event_clauses(segment.text), start=1)
        ]
        topic = detect_topic(segment.text)
        summary = summarize_text(segment.text, 120)
        importance = clamp01((score_result.final_score + min(1.0, len(entities) / 5)) / 2)
        filled_prompt = self.prompt_template.format(
            router_level=tier,
            information_score=f"{score_result.final_score:.4f}",
            segment_text=segment.text,
        )
        input_tokens = count_tokens(filled_prompt)
        output_tokens = max(64, count_tokens(summary) + len(facts) * 14 + len(episodes) * 16)
        latency_ms = max(80, 25 * count_tokens(segment.text))
        log_entry = self.cost_logger.log_extraction(
            segment_id=segment.segment_id,
            tier=tier,
            model_spec=model_spec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        return MemoryEntry(
            memory_id="",
            segment_id=segment.segment_id,
            topic=topic,
            summary=summary,
            semantic_memory=SemanticMemory(
                entities=entities,
                facts=facts,
                preferences=preferences,
                constraints=constraints,
            ),
            episodic_memory=EpisodicMemory(episodes=episodes),
            importance=importance,
            information_score=score_result.final_score,
            router_level=tier,
            extraction_mode="joint",
            extractor_name="mock_joint_extractor",
            model_used=model_spec.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=log_entry.cost_usd,
        )

    @staticmethod
    def _build_facts(text: str, entities: list[SemanticEntity]) -> list[SemanticFact]:
        topic = detect_topic(text)
        if entities:
            return [
                SemanticFact(
                    subject=entities[0].name,
                    predicate="related_to",
                    object=topic,
                    confidence=0.78,
                )
            ]
        return [
            SemanticFact(
                subject="conversation",
                predicate="topic",
                object=topic,
                confidence=0.65,
            )
        ]

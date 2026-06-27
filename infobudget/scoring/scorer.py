"""功能：聚合内在信息与记忆效用分数。
输入：Segment、配置与权重。
输出：ScoreResult。
依赖：配置、schemas、指标实现。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infobudget.config import ScoringConfig, WeightConfig
from infobudget.schemas import ScoreResult, Segment
from infobudget.scoring.intrinsic import (
    ConceptDensityScorer,
    EntropyScorer,
    EntityDensityScorer,
    LexicalDensityScorer,
)
from infobudget.scoring.utility import (
    EntityNoveltyScorer,
    EpisodicNoveltyScorer,
    MemorySearchable,
    SemanticNoveltyScorer,
)
from infobudget.utils.embeddings import HashingTextEncoder
from infobudget.utils.logging import get_logger
from infobudget.utils.text import clamp01

logger = get_logger(__name__)


@dataclass(slots=True)
class InformationScorer:
    """InfoBudget 第一阶段默认评分器。"""

    cfg: ScoringConfig
    weights: WeightConfig
    encoder: HashingTextEncoder | None = None
    entropy: EntropyScorer = field(init=False)
    lexical_density: LexicalDensityScorer = field(init=False)
    entity_density: EntityDensityScorer = field(init=False)
    concept_density: ConceptDensityScorer = field(init=False)
    semantic_novelty: SemanticNoveltyScorer = field(init=False)
    entity_novelty: EntityNoveltyScorer = field(init=False)
    episodic_novelty: EpisodicNoveltyScorer = field(init=False)

    def __post_init__(self) -> None:
        self.encoder = self.encoder or HashingTextEncoder()
        self.entropy = EntropyScorer()
        self.lexical_density = LexicalDensityScorer()
        self.entity_density = EntityDensityScorer()
        self.concept_density = ConceptDensityScorer()
        self.semantic_novelty = SemanticNoveltyScorer(self.encoder, self.cfg.novelty_top_k)
        self.entity_novelty = EntityNoveltyScorer(self.cfg.novelty_top_k)
        self.episodic_novelty = EpisodicNoveltyScorer(self.encoder, self.cfg.novelty_top_k)

    def score(self, segment: Segment, memory_store: MemorySearchable | None = None) -> ScoreResult:
        """计算最终路由分数。"""
        details = {
            "entropy": self.entropy.compute(segment.text),
            "lexical_density": self.lexical_density.compute(segment.text),
            "entity_density": self.entity_density.compute(segment.text),
            "concept_density": self.concept_density.compute(segment.text),
            "semantic_novelty": self.semantic_novelty.compute(segment, memory_store),
            "entity_novelty": self.entity_novelty.compute(segment, memory_store),
            "episodic_novelty": self.episodic_novelty.compute(segment, memory_store),
        }
        intrinsic_score = clamp01(
            details["entropy"] * self.weights.intrinsic.entropy
            + details["lexical_density"] * self.weights.intrinsic.lexical_density
            + details["entity_density"] * self.weights.intrinsic.entity_density
            + details["concept_density"] * self.weights.intrinsic.concept_density
        )
        utility_score = clamp01(
            details["semantic_novelty"] * self.weights.utility.semantic_novelty
            + details["entity_novelty"] * self.weights.utility.entity_novelty
            + details["episodic_novelty"] * self.weights.utility.episodic_novelty
        )
        final_score = clamp01(
            intrinsic_score * self.weights.fusion.intrinsic_weight
            + utility_score * self.weights.fusion.utility_weight
        )
        logger.info(
            "Scored segment %s -> intrinsic=%.3f utility=%.3f final=%.3f",
            segment.segment_id,
            intrinsic_score,
            utility_score,
            final_score,
        )
        return ScoreResult(
            intrinsic_score=intrinsic_score,
            utility_score=utility_score,
            final_score=final_score,
            details=details,
        )

"""功能：测试评分与路由模块。
输入：示例 segment。
输出：分数范围与 tier 断言。
依赖：pytest、评分与路由模块。
作者：OpenAI Codex
"""

from infobudget.config import load_project_bundle
from infobudget.routing.fixed_percentile import BudgetAwareRouter
from infobudget.schemas import Segment
from infobudget.scoring.intrinsic import ConceptDensityScorer
from infobudget.scoring.scorer import InformationScorer
from infobudget.scoring.utility import ActionabilityScorer
from infobudget.utils.text import extract_idea_units, tokenize_text


def test_scoring_and_routing() -> None:
    bundle = load_project_bundle("configs")
    scorer = InformationScorer(bundle.config.scoring, bundle.weights)
    router = BudgetAwareRouter(bundle.config.router.p33, bundle.config.router.p67)
    segment = Segment(
        segment_id="seg_000001",
        start_turn=1,
        end_turn=2,
        turn_ids=[1, 2],
        text="user: 我正在设计 InfoBudget，希望模块化实现 LiteTopicSeg 与 Router。",
        token_count=20,
        mean_adjacent_similarity=0.8,
        boundary_reason="start",
    )
    score = scorer.score(segment, memory_store=None)
    assert 0.0 <= score.final_score <= 1.0
    assert set(score.details) == {
        "entropy",
        "lexical_density",
        "entity_density",
        "concept_density",
        "information_gain",
        "actionability",
    }
    assert router.route(score.final_score) in {"small", "medium", "large"}


def test_scoring_mode_can_route_by_single_metric() -> None:
    bundle = load_project_bundle("configs")
    segment = Segment(
        segment_id="seg_000002",
        start_turn=1,
        end_turn=1,
        turn_ids=[1],
        text="Users must record final_score >= 0.71 and route to the large model.",
        token_count=12,
        mean_adjacent_similarity=0.0,
        boundary_reason="test",
    )

    full_score = InformationScorer(bundle.config.scoring, bundle.weights).score(segment, None)
    entropy_score = InformationScorer(
        bundle.config.scoring,
        bundle.weights,
        scoring_mode="entropy_only",
    ).score(segment, None)

    assert entropy_score.final_score == entropy_score.details["entropy"]
    assert full_score.final_score != entropy_score.final_score


def test_concept_density_uses_deduplicated_idea_units() -> None:
    text = (
        "Users want models to output route tables. "
        "Users want LLMs to output route tables."
    )
    units = extract_idea_units(text)

    assert units == [
        "language_model::output::route",
        "language_model::output::table",
        "output::route::table",
        "user::want::language_model",
        "user::want::output",
        "user::want::route",
    ]
    assert ConceptDensityScorer().compute(text) == len(units) / len(tokenize_text(text))


def test_actionability_scores_decision_rules_higher_than_weak_intent() -> None:
    scorer = ActionabilityScorer()
    strong = Segment(
        segment_id="strong",
        start_turn=1,
        end_turn=1,
        turn_ids=[1],
        text="When final_score >= 0.71, route the segment to the large model and record cost_log.",
        token_count=16,
        mean_adjacent_similarity=0.0,
        boundary_reason="test",
    )
    weak = Segment(
        segment_id="weak",
        start_turn=1,
        end_turn=1,
        turn_ids=[1],
        text="We could consider improving model routing later.",
        token_count=8,
        mean_adjacent_similarity=0.0,
        boundary_reason="test",
    )

    assert scorer.compute(strong, None) == 1.0
    assert scorer.compute(weak, None) < scorer.compute(strong, None)

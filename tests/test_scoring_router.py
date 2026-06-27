"""功能：测试评分与路由模块。
输入：示例 segment。
输出：分数范围与 tier 断言。
依赖：pytest、评分与路由模块。
作者：OpenAI Codex
"""

from infobudget.config import load_project_bundle
from infobudget.routing.fixed_percentile import BudgetAwareRouter
from infobudget.schemas import Segment
from infobudget.scoring.scorer import InformationScorer


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
    assert router.route(score.final_score) in {"small", "medium", "large"}

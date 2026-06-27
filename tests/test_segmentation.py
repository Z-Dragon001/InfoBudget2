"""功能：测试 LiteTopicSeg 行为。
输入：示例对话。
输出：分段断言。
依赖：pytest、项目分段模块。
作者：OpenAI Codex
"""

from infobudget.config import load_project_bundle
from infobudget.segmentation.lite_topic_seg import LiteTopicSeg
from infobudget.schemas import Turn
from infobudget.utils.text import count_tokens


def test_lite_topic_seg_produces_segments() -> None:
    bundle = load_project_bundle("configs")
    segmenter = LiteTopicSeg(bundle.config.segmentation)
    turns = [
        Turn(1, "user", "InfoBudget 关注长期记忆成本控制。", count_tokens("InfoBudget 关注长期记忆成本控制。")),
        Turn(2, "assistant", "我们先讨论分段和评分。", count_tokens("我们先讨论分段和评分。")),
        Turn(3, "user", "接下来切换到实验日志与成本统计设计。", count_tokens("接下来切换到实验日志与成本统计设计。")),
        Turn(4, "assistant", "最后再补测试。", count_tokens("最后再补测试。")),
    ]
    segments = segmenter.reindex_segments(segmenter.segment(turns))
    assert len(segments) >= 1
    assert segments[0].segment_id.startswith("seg_")

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


def test_lite_topic_seg_enriches_segment_text_with_blip_caption() -> None:
    bundle = load_project_bundle("configs")
    segmenter = LiteTopicSeg(bundle.config.segmentation)
    turns = [
        Turn(
            1,
            "user",
            "I visited the gallery today.",
            count_tokens("I visited the gallery today."),
            timestamp="2024-01-07T17:24:00.000",
            metadata={"blip_caption": "a mural with rainbow colors", "weekday": "Sun"},
        )
    ]
    segments = segmenter.reindex_segments(segmenter.segment(turns))
    assert "[2024-01-07T17:24:00.000, Sun] 0.user:" in segments[0].text
    assert "image description: a mural with rainbow colors" in segments[0].text

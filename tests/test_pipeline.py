"""功能：测试端到端 pipeline。
输入：示例 turns。
输出：memory、metrics 与输出文件断言。
依赖：pytest、项目 pipeline。
作者：OpenAI Codex
"""

from pathlib import Path

from infobudget.config import load_project_bundle
from infobudget.runtime.pipeline import InfoBudgetPipeline
from infobudget.schemas import Turn
from infobudget.utils.text import count_tokens


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    bundle = type(bundle)(
        root_dir=tmp_path,
        config=bundle.config,
        weights=bundle.weights,
        models=bundle.models,
        prices=bundle.prices,
        prompt_dir=bundle.prompt_dir,
    )
    pipeline = InfoBudgetPipeline(bundle)
    turns = [
        Turn(1, "user", "我在做一个叫 InfoBudget 的项目。", count_tokens("我在做一个叫 InfoBudget 的项目。")),
        Turn(2, "assistant", "它的核心目标是什么？", count_tokens("它的核心目标是什么？")),
        Turn(3, "user", "希望根据 segment 的信息价值决定使用什么模型提取长期记忆。", count_tokens("希望根据 segment 的信息价值决定使用什么模型提取长期记忆。")),
    ]
    result = pipeline.process_turns(turns, save_outputs=True)
    assert len(result.entries) == len(result.segments)
    assert all(entry.summary for entry in result.entries)
    assert (tmp_path / "outputs" / "metrics.json").exists()

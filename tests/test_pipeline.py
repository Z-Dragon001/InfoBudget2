"""功能：测试端到端 pipeline。
输入：示例 turns。
输出：memory、metrics 与输出文件断言。
依赖：pytest、项目 pipeline。
作者：OpenAI Codex
"""

from pathlib import Path
from dataclasses import replace

from infobudget.config import ProjectBundle, load_project_bundle
from infobudget.runtime.pipeline import InfoBudgetPipeline
from infobudget.schemas import Turn
from infobudget.utils.text import count_tokens


def _offline_bundle(bundle: ProjectBundle, root_dir: Path) -> ProjectBundle:
    config = replace(bundle.config, extractor=replace(bundle.config.extractor, mode="mock_joint"))
    return type(bundle)(
        root_dir=root_dir,
        config=config,
        weights=bundle.weights,
        models=bundle.models,
        prices=bundle.prices,
        prompt_dir=bundle.prompt_dir,
    )


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    bundle = _offline_bundle(bundle, tmp_path)
    pipeline = InfoBudgetPipeline(bundle)
    turns = [
        Turn(1, "user", "我在做一个叫 InfoBudget 的项目。", count_tokens("我在做一个叫 InfoBudget 的项目。")),
        Turn(2, "assistant", "它的核心目标是什么？", count_tokens("它的核心目标是什么？")),
        Turn(3, "user", "希望根据 segment 的信息价值决定使用什么模型提取长期记忆。", count_tokens("希望根据 segment 的信息价值决定使用什么模型提取长期记忆。")),
    ]
    result = pipeline.process_turns(turns, save_outputs=True)
    assert len(result.entries) >= len(result.segments)
    assert all(entry.memory for entry in result.entries)
    assert (tmp_path / "outputs" / "metrics.json").exists()


def test_pipeline_can_override_memory_output_root(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    bundle = _offline_bundle(bundle, tmp_path)
    run_output_dir = tmp_path / "outputs" / "memory" / "locomo" / "entropy_only" / "sample_1"
    pipeline = InfoBudgetPipeline(bundle, scoring_mode="entropy_only", run_output_dir=run_output_dir)
    turns = [
        Turn(1, "user", "InfoBudget should store memory outputs per scoring mode.", count_tokens("InfoBudget should store memory outputs per scoring mode.")),
    ]

    pipeline.process_turns(turns, save_outputs=False)
    pipeline.save_memory_outputs()

    assert (run_output_dir / "memory_jsonl" / "memory_entries.jsonl").exists()
    assert (run_output_dir / "memory_jsonl" / "segments.jsonl").exists()
    assert (run_output_dir / "memory_jsonl" / "cost_logs.jsonl").exists()
    assert (run_output_dir / "qdrant").exists()


def test_pipeline_loads_tier_specific_extraction_prompts(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    bundle = _offline_bundle(bundle, tmp_path)
    pipeline = InfoBudgetPipeline(bundle)

    small_prompt = pipeline.extractor.prompt_for_tier("small")
    medium_prompt = pipeline.extractor.prompt_for_tier("medium")
    large_prompt = pipeline.extractor.prompt_for_tier("large")

    assert "strict token budget" in small_prompt
    assert "Personal Information Extractor" in medium_prompt
    assert "high-recall Personal Information Extractor" in large_prompt

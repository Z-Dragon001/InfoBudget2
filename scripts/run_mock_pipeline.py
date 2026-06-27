"""功能：运行 InfoBudget 第一阶段 mock 端到端样例。
输入：本地配置与内置示例对话。
输出：outputs 下的日志与 memory 结果。
依赖：项目 pipeline。
作者：OpenAI Codex
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infobudget.runtime.pipeline import InfoBudgetPipeline
from infobudget.schemas import Turn
from infobudget.utils.text import count_tokens


def main() -> None:
    turns = [
        Turn(1, "user", "我在做一个叫 InfoBudget 的科研项目，目标是预算感知的长期记忆提取。", count_tokens("我在做一个叫 InfoBudget 的科研项目，目标是预算感知的长期记忆提取。")),
        Turn(2, "assistant", "这个系统要优先实现哪些模块？", count_tokens("这个系统要优先实现哪些模块？")),
        Turn(3, "user", "第一阶段先做 LiteTopicSeg、Information Scorer、P33/P67 Router、MockExtractor 和 Memory Store。", count_tokens("第一阶段先做 LiteTopicSeg、Information Scorer、P33/P67 Router、MockExtractor 和 Memory Store。")),
        Turn(4, "user", "另外要把 prompt、模型名和价格都配置化，不要写死。", count_tokens("另外要把 prompt、模型名和价格都配置化，不要写死。")),
    ]
    pipeline = InfoBudgetPipeline.from_config_dir("configs")
    result = pipeline.process_turns(turns, save_outputs=True)
    print(f"segments={len(result.segments)} memories={len(result.entries)} cost={result.metrics.total_cost_usd}")


if __name__ == "__main__":
    main()

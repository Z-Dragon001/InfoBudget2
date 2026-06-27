"""功能：导出运行时组件。
输入：运行时模块导入请求。
输出：pipeline 与注册表。
依赖：项目运行时模块。
作者：OpenAI Codex
"""

from infobudget.runtime.pipeline import InfoBudgetPipeline

__all__ = ["InfoBudgetPipeline"]

"""功能：导出评分组件。
输入：评分模块导入请求。
输出：评分器与指标类。
依赖：项目评分模块。
作者：OpenAI Codex
"""

from infobudget.scoring.scorer import InformationScorer

__all__ = ["InformationScorer"]

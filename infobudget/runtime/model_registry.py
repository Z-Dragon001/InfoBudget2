"""功能：提供模型注册表与价格注册表。
输入：配置中定义的模型与价格。
输出：模型查询与成本估算接口。
依赖：schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.schemas import ModelSpec, PriceSpec, Tier


@dataclass(slots=True)
class ModelRegistry:
    """tier 到模型的映射。"""

    models: dict[str, ModelSpec]

    def get(self, tier: Tier) -> ModelSpec:
        if tier not in self.models:
            raise KeyError(f"unknown tier: {tier}")
        return self.models[tier]


@dataclass(slots=True)
class PriceRegistry:
    """模型价格表。"""

    prices: dict[str, PriceSpec]

    def get(self, model_name: str) -> PriceSpec:
        if model_name not in self.prices:
            raise KeyError(f"missing price for model: {model_name}")
        return self.prices[model_name]

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        price = self.get(model_name)
        return (
            input_tokens / 1_000_000 * price.official_price_in_per_1m
            + output_tokens / 1_000_000 * price.official_price_out_per_1m
        )

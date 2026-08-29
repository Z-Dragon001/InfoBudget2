"""Lagrangian actor-critic training over prebuilt candidate memories."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from infobudget.rl_router.router import EmbeddingMLPRouter
from infobudget.rl_router.schemas import TIERS, Tier

Evaluator = Callable[[list[Tier]], tuple[float, float]]


@dataclass(slots=True)
class TrainingStep:
    qa_score: float
    virtual_cost: float
    reward: float
    lagrange_multiplier: float
    actions: list[Tier]


class ConstrainedActorCriticTrainer:
    def __init__(
        self,
        model: EmbeddingMLPRouter,
        *,
        budget: float,
        learning_rate: float = 3e-4,
        lambda_learning_rate: float = 1e-2,
        value_loss_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_gradient_norm: float = 1.0,
        seed: int = 42,
    ):
        self.model = model
        self.budget = float(budget)
        self.lambda_learning_rate = float(lambda_learning_rate)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_gradient_norm = float(max_gradient_norm)
        self.lagrange_multiplier = 0.0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def step(self, features: np.ndarray, evaluate: Evaluator) -> TrainingStep:
        self.model.train()
        tensor = torch.as_tensor(features, dtype=torch.float32, device=next(self.model.parameters()).device)
        logits, values = self.model(tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        tiers = [TIERS[int(index)] for index in actions.detach().cpu()]
        qa_score, virtual_cost = evaluate(tiers)
        reward = float(qa_score - self.lagrange_multiplier * (virtual_cost - self.budget))
        target = torch.full_like(values, reward)
        advantage = (target - values).detach()
        policy_loss = -(distribution.log_prob(actions) * advantage).mean()
        value_loss = torch.nn.functional.mse_loss(values, target)
        entropy_bonus = distribution.entropy().mean()
        loss = (
            policy_loss
            + self.value_loss_coefficient * value_loss
            - self.entropy_coefficient * entropy_bonus
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
        self.optimizer.step()
        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier + self.lambda_learning_rate * (virtual_cost - self.budget),
        )
        return TrainingStep(qa_score, virtual_cost, reward, self.lagrange_multiplier, tiers)

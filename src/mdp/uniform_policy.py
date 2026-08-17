from __future__ import annotations

from dataclasses import dataclass

import torch
from rl_commons.policies.policy import Policy


@dataclass
class UniformPolicyConfig:
    action_range: tuple[float, float] = (-1.0, 1.0)


class UniformPolicy(Policy):
    """Policy not backed by a network. Ignores the observation and samples uniformly over the action range."""

    def __init__(self, input_size : int, number_actions : int, config : UniformPolicyConfig = UniformPolicyConfig()):
        super().__init__(input_size, number_actions)
        self.config = config
        self.action_low = float(config.action_range[0])
        self.action_high = float(config.action_range[1])

    def forward(self, observation : torch.Tensor) -> torch.distributions.Distribution:
        batch_shape = observation.shape[:-1]
        low = torch.full((*batch_shape, self._number_actions), self.action_low,
                          device=observation.device, dtype=observation.dtype)
        high = torch.full((*batch_shape, self._number_actions), self.action_high,
                           device=observation.device, dtype=observation.dtype)
        return torch.distributions.Uniform(low, high)

    def log_probability(self, action : torch.Tensor, distribution : torch.distributions.Distribution) -> torch.Tensor:
        return distribution.log_prob(action).sum(-1, keepdim=True)


def _build_uniform_policy(obs_dimension: int, action_dimension: int, config) -> Policy:
    cfg = config if isinstance(config, UniformPolicyConfig) else UniformPolicyConfig()
    return UniformPolicy(obs_dimension, action_dimension, cfg)


Policy.register('uniform', _build_uniform_policy)

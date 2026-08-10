from dataclasses import dataclass, field
from typing import Literal

import torch

ActivationId = Literal["relu", "tanh", "leaky_relu"]

ACTIVATION_MAP: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "leaky_relu": torch.nn.LeakyReLU,
}


@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: ActivationId = "relu"

    def build_trunk(self, input_size: int) -> tuple[torch.nn.Sequential, int]:
        """Build a MLP trunk (no output head). Returns (sequential, output_size)."""
        activation_function = ACTIVATION_MAP[self.activation]
        layers: list[torch.nn.Module] = []
        in_size = input_size
        for h in self.hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(activation_function())
            in_size = h
        return torch.nn.Sequential(*layers), in_size
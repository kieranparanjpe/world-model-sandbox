import torch
from torch import nn

from src.algorithms.jepa.encoder_config import EncoderConfig


class Encoder(nn.Module):
    input_size: int
    encoding_space_size: int

    def __init__(self, input_size: int, config: EncoderConfig):
        super().__init__()
        self.input_size = int(input_size)
        self.encoding_space_size = int(config.encoding_space_size)

        trunk, out_size = config.build_trunk(self.input_size)
        self._net = trunk
        self._head = torch.nn.Linear(int(out_size), self.encoding_space_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._head(self._net(observation))

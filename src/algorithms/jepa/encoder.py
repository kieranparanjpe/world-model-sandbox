import torch
from torch import nn

from src.algorithms.jepa.encoder_config import EncoderConfig


class Encoder(nn.Module):

    def __init__(self, input_size: int, config: EncoderConfig):
        super().__init__()
        trunk, out_size = config.build_trunk(input_size)
        self._net = trunk
        self._head = torch.nn.Linear(out_size, config.encoding_space_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._head(self._net(observation))

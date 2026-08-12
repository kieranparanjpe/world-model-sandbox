import torch
from torch import nn

from src.algorithms.jepa_decoder.decoder_config import DecoderConfig


class Decoder(nn.Module):
    output_size: int
    encoding_space_size: int

    def __init__(self, output_size: int, config: DecoderConfig):
        super().__init__()
        self.output_size = int(output_size)
        self.encoding_space_size = int(config.encoding_space_size)

        trunk, out_size = config.build_trunk(self.encoding_space_size)
        self._net = trunk
        self._head = torch.nn.Linear(int(out_size), self.output_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._head(self._net(observation))

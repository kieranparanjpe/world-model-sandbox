import torch
from torch import nn

from src.algorithms.jepa.predictor_config import PredictorConfig


class Predictor(nn.Module):
    action_size: int
    config: PredictorConfig

    def __init__(self, action_size : int, config: PredictorConfig):
        super().__init__()
        self.action_size = int(action_size)
        self.config = config
        trunk, out_size = config.build_trunk(int(self.config.encoding_space_size) + self.action_size)
        self._net = trunk
        self._head = torch.nn.Linear(int(out_size), int(self.config.encoding_space_size))

    def forward(self, encoding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoding_action = torch.concat((encoding, action), dim=-1)
        return self._head(self._net(encoding_action))

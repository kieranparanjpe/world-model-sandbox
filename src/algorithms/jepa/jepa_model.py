import torch
import torch.nn as nn

from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.predictor import Predictor


class JEPAModel(nn.Module):
    def __init__(self, encoder : Encoder, predictor : Predictor):
        super().__init__()

        self.encoder : Encoder = encoder
        self.predictor : Predictor = predictor

    def forward(self, observation : torch.Tensor, action : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.encoder(observation)

        prediction = self.predictor(encoding, action)

        return prediction, encoding
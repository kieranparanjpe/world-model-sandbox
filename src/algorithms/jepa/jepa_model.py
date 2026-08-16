from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from ml_commons.networks import SaveableNetwork

from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.predictor import Predictor


class JEPAModel(SaveableNetwork, nn.Module):
    def __init__(self, encoder : Encoder, predictor : Predictor):
        super().__init__()

        self.encoder : Encoder = encoder
        self.predictor : Predictor = predictor

    def forward(self, observation : torch.Tensor, action : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.encoder(observation)

        prediction = self.predictor(encoding, action)

        return prediction, encoding

    def save(self, path, norm_stats=None):
        save_dict = {
            "model": self.state_dict(),
            "encoder_config": self.encoder.config,
            "predictor_config": self.predictor.config,
            "input_size": self.encoder.input_size,
            "action_size": self.predictor.action_size,
        }
        if norm_stats is not None:
            save_dict["norm_stats"] = norm_stats
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kwargs) -> tuple[JEPAModel, Optional[dict]]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True) if path else {}

        encoder = Encoder(checkpoint["input_size"], checkpoint["encoder_config"])
        predictor = Predictor(checkpoint["action_size"], checkpoint["predictor_config"])
        model = JEPAModel(encoder, predictor)

        model.load_state_dict(checkpoint["model"])
        model.eval()

        return model, checkpoint.get("norm_stats")
from __future__ import annotations

import torch
import torch.nn as nn
from ml_commons.networks import SaveableNetwork
from ml_commons.stats import NormalisationStats

from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.predictor import Predictor


class JEPAModel(SaveableNetwork, nn.Module):
    obs_norm_stats: NormalisationStats
    action_norm_stats: NormalisationStats

    def __init__(self, encoder : Encoder, predictor : Predictor):
        super().__init__()

        self.encoder : Encoder = encoder
        self.predictor : Predictor = predictor
        self.obs_norm_stats = NormalisationStats()
        self.action_norm_stats = NormalisationStats()

    def forward(self, observation : torch.Tensor, action : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.encoder(observation)

        prediction = self.predictor(encoding, action)

        return prediction, encoding

    def save(self, path):
        save_dict = {
            "model": self.state_dict(),
            "encoder_config": self.encoder.config,
            "predictor_config": self.predictor.config,
            "input_size": self.encoder.input_size,
            "action_size": self.predictor.action_size,
            "obs_norm_stats": self.obs_norm_stats,
            "action_norm_stats": self.action_norm_stats,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kwargs) -> JEPAModel:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True) if path else {}

        encoder = Encoder(checkpoint["input_size"], checkpoint["encoder_config"])
        predictor = Predictor(checkpoint["action_size"], checkpoint["predictor_config"])
        model = JEPAModel(encoder, predictor)

        model.load_state_dict(checkpoint["model"])
        model.obs_norm_stats = checkpoint.get("obs_norm_stats", NormalisationStats())
        model.action_norm_stats = checkpoint.get("action_norm_stats", NormalisationStats())
        model.to(map_location)
        model.eval()

        return model
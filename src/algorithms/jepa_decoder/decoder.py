from __future__ import annotations

import torch
from torch import nn
from ml_commons.networks import SaveableNetwork
from ml_commons.stats import NormalisationStats

from src.algorithms.jepa_decoder.decoder_config import DecoderConfig

class Decoder(SaveableNetwork, nn.Module):
    output_size: int
    config: DecoderConfig
    obs_norm_stats: NormalisationStats

    def __init__(self, output_size: int, config: DecoderConfig):
        super().__init__()
        self.output_size = int(output_size)
        self.config = config
        self.obs_norm_stats = NormalisationStats()

        trunk, out_size = config.build_trunk(int(self.config.encoding_space_size))
        self._net = trunk
        self._head = torch.nn.Linear(int(out_size), self.output_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self._head(self._net(observation))

    def save(self, path):
        save_dict = {
            "model": self.state_dict(),
            "config": self.config,
            "output_size": self.output_size,
            "obs_norm_stats": self.obs_norm_stats,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kwargs) -> Decoder:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True) if path else {}
        decoder: Decoder = Decoder(checkpoint["output_size"], checkpoint["config"])

        state_dict = checkpoint["model"]
        decoder.load_state_dict(state_dict)
        decoder.obs_norm_stats = checkpoint.get("obs_norm_stats", NormalisationStats())

        decoder.eval()

        return decoder

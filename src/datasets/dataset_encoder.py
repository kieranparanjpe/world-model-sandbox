import pathlib
from typing import Optional

import torch
from ml_commons.stats import NormalisationStats
from torch.utils.data import Dataset


class DatasetEncoder(Dataset):

    def __init__(self, path : pathlib.Path, obs_norm : Optional[NormalisationStats] = None):
        dataset_dict = torch.load(path, weights_only=True)
        self.encodings : torch.Tensor = dataset_dict["encodings"].detach()
        self.raw_observations : torch.Tensor = dataset_dict["raw_observations"].detach()

        if len(self.encodings) != len(self.raw_observations):
            raise ValueError("X and Y length mismatch")

        self._obs_norm : Optional[NormalisationStats] = None
        if obs_norm is not None:
            self.apply_obs_normalization(obs_norm)

    def apply_obs_normalization(self, norm : NormalisationStats):
        mean, std = norm.as_tensors(dtype=self.raw_observations.dtype)
        self.raw_observations.subtract_(mean).divide_(std)
        self._obs_norm = norm

    def get_norm_stats(self) -> dict[str, NormalisationStats]:
        return {"obs": self._obs_norm if self._obs_norm is not None else self._compute_obs_stats()}

    def _compute_obs_stats(self) -> NormalisationStats:
        return NormalisationStats(
            mean=self.raw_observations.mean(dim=0).numpy(),
            var=self.raw_observations.var(dim=0).numpy() + 1e-8,
        )

    def __len__(self):
        return len(self.raw_observations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {
            "encodings": self.encodings[idx],
            "raw_observations": self.raw_observations[idx],
        }


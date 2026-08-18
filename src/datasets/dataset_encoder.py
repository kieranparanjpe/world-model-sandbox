import pathlib
from typing import Optional

import torch
from ml_commons.stats import NormalisationStats
from torch.utils.data import Dataset

from src.datasets.norm_stats_keys import OBS_NORM_KEY


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
        if self._obs_norm is not None:
            raise ValueError("Observation normalization has already been applied to this dataset.")
        mean, std = norm.as_tensors(dtype=self.raw_observations.dtype)
        self.raw_observations.subtract_(mean).divide_(std)
        self._obs_norm = norm

    def normalise_obs(self):
        """Compute obs stats from this dataset's current (raw) data, store them, and apply."""
        self.apply_obs_normalization(self._compute_obs_stats())

    def get_norm_stats(self) -> dict[str, NormalisationStats]:
        """Pure getter -- whatever's been applied, or identity if nothing has. Never computes."""
        return {OBS_NORM_KEY: self._obs_norm if self._obs_norm is not None else NormalisationStats()}

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


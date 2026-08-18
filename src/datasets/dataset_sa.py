import pathlib
from typing import Optional

import torch
from ml_commons.stats import NormalisationStats
from torch.utils.data import Dataset

from src.datasets.norm_stats_keys import OBS_NORM_KEY, ACTION_NORM_KEY


class DatasetSA(Dataset):

    def __init__(self, path : pathlib.Path,
                 obs_norm : Optional[NormalisationStats] = None,
                 action_norm : Optional[NormalisationStats] = None):
        dataset_dict = torch.load(path, weights_only=True)
        self.current_observations : torch.Tensor = dataset_dict["current_observations"].detach()
        self.actions : torch.Tensor = dataset_dict["actions"].detach()
        self.next_observations : torch.Tensor = dataset_dict["next_observations"].detach()

        if len(self.current_observations) != len(self.actions) != len(self.next_observations):
            raise ValueError("Observation and action length mismatch.")

        self._obs_norm : Optional[NormalisationStats] = None
        self._action_norm : Optional[NormalisationStats] = None

        if obs_norm is not None:
            self.apply_obs_normalization(obs_norm)
        if action_norm is not None:
            self.apply_action_normalization(action_norm)

    def apply_obs_normalization(self, norm : NormalisationStats):
        if self._obs_norm is not None:
            raise ValueError("Observation normalization has already been applied to this dataset.")
        mean, std = norm.as_tensors(dtype=self.current_observations.dtype)
        self.current_observations.subtract_(mean).divide_(std)
        self.next_observations.subtract_(mean).divide_(std)
        self._obs_norm = norm

    def apply_action_normalization(self, norm : NormalisationStats):
        if self._action_norm is not None:
            raise ValueError("Action normalization has already been applied to this dataset.")
        mean, std = norm.as_tensors(dtype=self.actions.dtype)
        self.actions.subtract_(mean).divide_(std)
        self._action_norm = norm

    def normalise_obs(self):
        """Compute obs stats from this dataset's current (raw) data, store them, and apply."""
        self.apply_obs_normalization(self._compute_obs_stats())

    def normalise_actions(self):
        """Compute action stats from this dataset's current (raw) data, store them, and apply."""
        self.apply_action_normalization(self._compute_action_stats())

    def get_norm_stats(self) -> dict[str, NormalisationStats]:
        """Pure getter -- whatever's been applied, or identity if nothing has. Never computes."""
        return {
            OBS_NORM_KEY: self._obs_norm if self._obs_norm is not None else NormalisationStats(),
            ACTION_NORM_KEY: self._action_norm if self._action_norm is not None else NormalisationStats(),
        }

    def _compute_obs_stats(self) -> NormalisationStats:
        return NormalisationStats(
            mean=self.current_observations.mean(dim=0).numpy(),
            var=self.current_observations.var(dim=0).numpy() + 1e-8,
        )

    def _compute_action_stats(self) -> NormalisationStats:
        return NormalisationStats(
            mean=self.actions.mean(dim=0).numpy(),
            var=self.actions.var(dim=0).numpy() + 1e-8,
        )

    def __len__(self):
        return len(self.current_observations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {
            "current_observations": self.current_observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx]
        }


import pathlib

import torch
from torch.utils.data import Dataset


class DatasetSA(Dataset):

    def __init__(self, path : pathlib.Path):
        dataset_dict = torch.load(path, weights_only=True)
        self.current_observations : torch.Tensor = dataset_dict["current_observations"]
        self.actions : torch.Tensor = dataset_dict["actions"]
        self.next_observations : torch.Tensor = dataset_dict["next_observations"]

        if len(self.current_observations) != len(self.actions) != len(self.next_observations):
            raise ValueError("Observation and action length mismatch.")

    def normalise_actions(self):
        self.actions.subtract_(self.actions.mean(dim=0)).divide_(self.actions.std(dim=0) + 1e-8)

    def normalise_observations(self):
        offset = self.current_observations.mean(dim=0)
        scale = self.current_observations.std(dim=0) + 1e-8

        self.current_observations.subtract_(offset).divide_(scale)
        self.next_observations.subtract_(offset).divide_(scale)

    def __len__(self):
        return len(self.current_observations)

    def __getitem__(self, idx):
        return {
            "current_observation": self.current_observations[idx],
            "action": self.actions[idx],
            "next_observation": self.next_observations[idx]
        }


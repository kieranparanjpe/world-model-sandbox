import pathlib

import torch
from torch.utils.data import Dataset


class DatasetEncoder(Dataset):

    def __init__(self, path : pathlib.Path):
        dataset_dict = torch.load(path, weights_only=True)
        self.encodings : torch.Tensor = dataset_dict["encodings"].detach()
        self.raw_observations : torch.Tensor = dataset_dict["raw_observations"].detach()

        if len(self.encodings) != len(self.raw_observations):
            raise ValueError("X and Y length mismatch")

    def normalise_observations(self):
        offset = self.raw_observations.mean(dim=0)
        scale = self.raw_observations.std(dim=0) + 1e-8

        self.raw_observations.subtract_(offset).divide_(scale)

    def __len__(self):
        return len(self.raw_observations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {
            "encodings": self.encodings[idx],
            "raw_observations": self.raw_observations[idx],
        }


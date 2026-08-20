import argparse
from datetime import datetime
from typing import Optional

import torch.distributions
from ml_commons.stats import NormalisationStats
from rl_commons.mdp import Mdp, MdpTerminationState, MdpGym
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import torch

from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.jepa_model import JEPAModel
from src.datasets.dataset_sa import DatasetSA


class DatasetFromEncoder:

    def __init__(self, dataset_path : str, model_path : str, task_id : str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = JEPAModel.load(model_path, map_location=self.device)

        self._encoder = model.encoder.to(self.device)
        self._datasetSA = DatasetSA(Path(dataset_path))
        self._obs_norm = model.obs_norm_stats

        time = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
        self._output_path = Path(__file__).resolve().parents[2] / "datasets" / task_id / "decoder" / f"dataset_{time}.pt"

        self._dataset = {
            "encodings" : torch.zeros((len(self._datasetSA), self._encoder.config.encoding_space_size), dtype=torch.float32,
                                                 device=self.device),
            "raw_observations": torch.zeros((len(self._datasetSA), self._encoder.input_size), dtype=torch.float32,
                                                 device=self.device),
            "dataset_path": dataset_path,
            "model_path": model_path,
        }

    def collect(self):
        batch_size = 512
        dataloader = DataLoader(self._datasetSA, batch_size=batch_size)

        mean, std = None, None
        if self._obs_norm is not None:
            mean, std = self._obs_norm.as_tensors(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                observations = batch["current_observations"].to(self.device)
                encoder_input = observations if mean is None else (observations - mean) / std
                encodings = self._encoder(encoder_input)
                start = idx * batch_size
                end = min(start + batch_size, len(self._datasetSA))
                self._dataset["encodings"][start : end] = encodings
                # raw_observations stays unnormalized -- DatasetEncoder is the single point stats get applied
                self._dataset["raw_observations"][start: end] = observations

    def save(self):
        file_path = Path(self._output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._dataset, file_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Task id", default="CartPole-v1")
    parser.add_argument("--dataset", "-d", help="Path to dataset SA")
    parser.add_argument("--model", "-m", help="Path to saved jepa model")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_from_encoder = DatasetFromEncoder(args.dataset, args.model, args.environment)
    dataset_from_encoder.collect()
    dataset_from_encoder.save()


if __name__ == "__main__":
    main()
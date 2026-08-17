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

    def __init__(self, datasetSA : DatasetSA, encoder : Encoder, task_id : str,
                 obs_norm : Optional[NormalisationStats] = None):
        self._encoder = encoder
        self._datasetSA = datasetSA
        self._obs_norm = obs_norm

        time = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
        self._output_path = Path(__file__).resolve().parents[2] / "datasets" / task_id / "decoder" / f"dataset_{time}.pt"

        self._dataset = {
            "encodings" : torch.zeros((len(self._datasetSA), self._encoder.config.encoding_space_size), dtype=torch.float32,
                                                 device=torch.device("cpu")),
            "raw_observations": torch.zeros((len(self._datasetSA), self._encoder.input_size), dtype=torch.float32,
                                                 device=torch.device("cpu")),
        }

    def collect(self):
        batch_size = 512
        dataloader = DataLoader(self._datasetSA, batch_size=batch_size)

        mean, std = None, None
        if self._obs_norm is not None:
            mean, std = self._obs_norm.as_tensors(dtype=torch.float32)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                observations = batch["current_observations"]
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

    datasetSA = DatasetSA(args.dataset)
    model, norm_stats = JEPAModel.load(args.model)

    dataset_from_encoder = DatasetFromEncoder(datasetSA, model.encoder, args.environment,
                                              obs_norm=norm_stats.get("obs") if norm_stats else None)
    dataset_from_encoder.collect()
    dataset_from_encoder.save()


if __name__ == "__main__":
    main()
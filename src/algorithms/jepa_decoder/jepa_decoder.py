import copy
import os
from typing import Optional, Callable

import torch
from ml_commons.config import RunInfo
from ml_commons.log import Logger, NullLogger
from torch import nn, optim
from torch.utils.data import DataLoader

from src.algorithms import Algorithm

from src.algorithms.jepa_decoder.decoder import Decoder
from src.algorithms.jepa_decoder.jepa_decoder_config import JepaDecoderConfig
from src.algorithms.utils import save_model
from src.datasets.dataset_encoder import DatasetEncoder
from src.datasets.dataset_sa import DatasetSA

def jepa_decoder_factory(hyperparameters : JepaDecoderConfig,
                 run_info : RunInfo,
                 obs_dimension : int,
                 dataset : DatasetEncoder,
                 logger : Optional[Logger],
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
    decoder_factory = lambda: Decoder(obs_dimension, hyperparameters.decoder_config)
    return JEPADecoder(hyperparameters, run_info, obs_dimension, decoder_factory, dataset, logger, device,
                should_save_models)


class JEPADecoder(Algorithm):
    decoder: Decoder

    decoder_optimiser: torch.optim.Optimizer

    hyperparameters: JepaDecoderConfig
    dataset: DatasetEncoder

    def __init__(self, hyperparameters : JepaDecoderConfig,
                 run_info : RunInfo,
                 obs_dimension: int,
                 decoder_factory : Callable[[], Decoder],
                 dataset : DatasetEncoder,
                 logger: Logger = NullLogger(),
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
        super().__init__(hyperparameters, run_info, obs_dimension, dataset, logger, device, should_save_models)

        self.decoder_factory = decoder_factory
        self.reset_models()

        self.criterion = nn.MSELoss()

    # noinspection PyAttributeOutsideInit
    def reset_models(self):
        self.decoder = self.decoder_factory()

        self.decoder_optimiser = optim.Adam(self.decoder.parameters(),
                                            lr=self.hyperparameters.decoder_lr,
                                            weight_decay=self.hyperparameters.decoder_regularization)

    def save_models(self, current_epoch : int):
        model_path = self.run_info.local_folder_path("saved_networks/jepa/decoder")
        os.makedirs(model_path, exist_ok=True)

        width = len(str(self.hyperparameters.epochs))

        self.decoder.save(f'{model_path}/decoder_{current_epoch:0{width}d}.pt',
                          norm_stats=self.dataset.get_norm_stats())

    def train_single_epoch(self, train_loader : DataLoader[DatasetEncoder]):
        """
        Side effect: places loss into logger["losses/loss"]
        """
        self.decoder.train()

        self.logger.reset("losses/train_loss")

        for batch in train_loader:
            raw_observations, encodings = batch["raw_observations"].to(self.device), batch["encodings"].to(self.device)

            predicted = self.decoder(encodings)

            loss = self.criterion(predicted, raw_observations)

            self.decoder.zero_grad()

            loss.backward()

            self.decoder_optimiser.step()

            with torch.no_grad():
                # Logging and stats
                batch_length = raw_observations.size(0)
                total_samples = len(train_loader.dataset) # type: ignore
                self.logger.sum_log_data({
                    "losses/train_loss": loss.item() * batch_length / total_samples
                })

    def evaluate(self, validation_loader : DataLoader[DatasetEncoder]):
        self.decoder.eval()

        self.logger.reset("losses/validation_loss")

        with torch.no_grad():
            for batch in validation_loader:
                raw_observations, encodings = batch["raw_observations"].to(self.device), batch["encodings"].to(
                    self.device)

                predicted = self.decoder(encodings)
                loss = self.criterion(predicted, raw_observations)

                # Logging and stats
                batch_length = raw_observations.size(0)
                total_samples = len(validation_loader.dataset)  # type: ignore
                self.logger.sum_log_data({
                    "losses/validation_loss": loss.item() * batch_length / total_samples
                })



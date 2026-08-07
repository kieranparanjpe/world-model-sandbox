import copy
from typing import Optional

import torch
from rl_commons.log import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.algorithms import Algorithm
from src.algorithms.algorithm_config import JepaConfig
from src.datasets.dataset_sa import DatasetSA


class JEPA(Algorithm):

    def __init__(self, hyperparameters : JepaConfig,
                 obs_dimension: int,
                 encoder : nn.Module,
                 predictor : nn.Module,
                 dataset : DatasetSA,
                 logger: Optional[Logger] = None,
                 device: torch.device = torch.device('cpu')):
        super().__init__(hyperparameters, obs_dimension, dataset, logger, device)

        self.encoder = encoder
        self.predictor = predictor
        self.label_encoder = copy.deepcopy(encoder)
        self.hyperparameters : JepaConfig

        self.criterion = nn.MSELoss()
        self.encoder_optimiser = optim.Adam(self.encoder.parameters(),
                                            lr=self.hyperparameters.encoder_lr,
                                            weight_decay=self.hyperparameters.encoder_regularization)
        self.predictor_optimiser = optim.Adam(self.predictor.parameters(),
                                              lr=self.hyperparameters.predictor_lr,
                                              weight_decay=self.hyperparameters.predictor_regularization)

        self.logger.add_elements({
            "losses/loss": 0.0,
        })

        # Freeze grads for label encoder. it is updated with EMA
        for param in self.label_encoder.parameters():
            param.requires_grad = False

    def resolve_dataset(self, dataset : DatasetSA):
        pass

    def train_single_epoch(self, train_loader : DataLoader[DatasetSA]):

        for batch in train_loader:
            batch : dict[str, torch.Tensor]
            current_observations = batch["current_observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            next_observations = batch["next_observations"].to(self.device)

            encoding = self.encoder(current_observations)
            prediction = self.predictor(encoding, actions)

            encoded_label = self.label_encoder(next_observations)

            loss = self.criterion(prediction, encoded_label)

            self.encoder_optimiser.zero_grad()
            self.predictor_optimiser.zero_grad()

            loss.backward()

            self.encoder_optimiser.step()
            self.predictor_optimiser.step()

            # Logging and stats
            with torch.no_grad():
                batch_length = current_observations.size(0)
                total_samples = len(train_loader.dataset) # type: ignore
                self.logger.sum_log_data({
                    "losses/loss": loss.item() * batch_length / total_samples
                })

        self.logger.log_data("losses/loss", "epoch")
        self.logger.reset("losses/loss")


    def train(self):

        for epoch in tqdm(range(self.hyperparameters.epochs)):
            self.train_single_epoch(DataLoader(self.dataset))
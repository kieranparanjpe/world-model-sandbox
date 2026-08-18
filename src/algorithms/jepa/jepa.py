import copy
import os
from typing import Optional, Callable

import numpy as np
import torch
from ml_commons.config import RunInfo
from ml_commons.log import Logger
from torch import nn, optim
from torch.utils.data import DataLoader

from src.algorithms import Algorithm
from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.jepa_config import JepaConfig
from src.algorithms.jepa.jepa_model import JEPAModel
from src.algorithms.jepa.predictor import Predictor
from src.algorithms.utils import save_model
from src.datasets.dataset_sa import DatasetSA
from src.datasets.norm_stats_keys import OBS_NORM_KEY, ACTION_NORM_KEY

def jepa_factory(hyperparameters : JepaConfig,
                 run_info : RunInfo,
                 obs_dimension : int,
                 action_dimension : int,
                 dataset : DatasetSA,
                 logger : Optional[Logger],
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
    encoder_factory = lambda : Encoder(obs_dimension, hyperparameters.encoder_config)
    predictor_factory = lambda : Predictor(action_dimension, hyperparameters.predictor_config)

    return JEPA(hyperparameters, run_info, obs_dimension, encoder_factory, predictor_factory, dataset, logger, device,
                should_save_models)


class JEPA(Algorithm):
    model: JEPAModel
    label_encoder: Encoder

    encoder_optimiser: torch.optim.Optimizer
    predictor_optimiser: torch.optim.Optimizer

    hyperparameters: JepaConfig
    dataset: DatasetSA

    def __init__(self, hyperparameters : JepaConfig,
                 run_info : RunInfo,
                 obs_dimension: int,
                 encoder_factory : Callable[[], Encoder],
                 predictor_factory : Callable[[], Predictor],
                 dataset : DatasetSA,
                 logger: Optional[Logger] = None,
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
        super().__init__(hyperparameters, run_info, obs_dimension, dataset, logger, device, should_save_models)

        self.encoder_factory = encoder_factory
        self.predictor_factory = predictor_factory
        self.reset_models()

        self.criterion = nn.MSELoss()

    # noinspection PyAttributeOutsideInit
    def reset_models(self):
        self.model = JEPAModel(self.encoder_factory(), self.predictor_factory())
        self.label_encoder = copy.deepcopy(self.model.encoder)

        self.encoder_optimiser = optim.Adam(self.model.encoder.parameters(),
                                            lr=self.hyperparameters.encoder_lr,
                                            weight_decay=self.hyperparameters.encoder_regularization)
        self.predictor_optimiser = optim.Adam(self.model.predictor.parameters(),
                                              lr=self.hyperparameters.predictor_lr,
                                              weight_decay=self.hyperparameters.predictor_regularization)

        # Freeze grads for label encoder. it is updated with EMA
        for param in self.label_encoder.parameters():
            param.requires_grad = False

    def save_models(self, current_epoch : int):
        model_path = self.run_info.local_folder_path("saved_networks/jepa/model")
        os.makedirs(model_path, exist_ok=True)

        width = len(str(self.hyperparameters.epochs))

        norm_stats = self.dataset.get_norm_stats()
        self.model.obs_norm_stats = norm_stats[OBS_NORM_KEY]
        self.model.action_norm_stats = norm_stats[ACTION_NORM_KEY]
        self.model.save(f'{model_path}/model_{current_epoch:0{width}d}.pt')

    def forward(self, current_observations, actions, next_observations):
        prediction, encoding = self.model.forward(current_observations, actions)

        encoded_label = self.label_encoder(next_observations)

        loss = self.criterion(prediction, encoded_label)

        return encoding, prediction, encoded_label, loss

    def extract_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (batch["current_observations"].to(self.device),
                batch["actions"].to(self.device),
                batch["next_observations"].to(self.device))

    def train_single_epoch(self, train_loader : DataLoader[DatasetSA]):
        """
        Side effect: places loss into logger["losses/loss"]
        """
        self.model.train()
        self.label_encoder.train()

        self.logger.reset("losses/train_loss")

        for batch in train_loader:
            current_observations, actions, next_observations = self.extract_batch(batch)

            _, _, _, loss = self.forward(current_observations, actions, next_observations)

            self.model.zero_grad()

            loss.backward()

            self.encoder_optimiser.step()
            self.predictor_optimiser.step()

            with torch.no_grad():
                # Update label encoder weights with EMA
                for input_parameter, label_parameter in zip(self.label_encoder.parameters(),
                                                            self.model.encoder.parameters()):
                    # Lerp is the same as EMA
                    label_parameter.lerp_(input_parameter, self.hyperparameters.label_encoder_ema_momentum)

                # Logging and stats
                batch_length = current_observations.size(0)
                total_samples = len(train_loader.dataset) # type: ignore
                self.logger.sum_log_data({
                    "losses/train_loss": loss.item() * batch_length / total_samples
                })

    def evaluate(self, validation_loader : DataLoader[DatasetSA]):
        self.model.eval()
        self.label_encoder.eval()

        self.logger.reset("losses/validation_loss")

        with torch.no_grad():
            for batch in validation_loader:
                current_observations, actions, next_observations = self.extract_batch(batch)
                _, _, _, loss = self.forward(current_observations, actions, next_observations)

                # Logging and stats
                batch_length = current_observations.size(0)
                total_samples = len(validation_loader.dataset)  # type: ignore
                self.logger.sum_log_data({
                    "losses/validation_loss": loss.item() * batch_length / total_samples
                })



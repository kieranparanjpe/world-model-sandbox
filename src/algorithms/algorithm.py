from abc import ABC, abstractmethod
from typing import Optional, Sized, cast

import numpy as np
import torch
from ml_commons.config import RunInfo
from ml_commons.log import Logger, NullLogger, WandBLogger
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm.auto import tqdm

from src.algorithms.algorithm_config import AlgorithmConfig
from src.algorithms.validation_method_config import KFoldValidationConfig, TrainValidationConfig


class Algorithm(ABC):

    def __init__(self, hyperparameters : AlgorithmConfig,
                 run_info : RunInfo,
                 obs_dimension: int,
                 dataset : Dataset,
                 logger: Logger = NullLogger(),
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.run_info = run_info
        self.obs_dimension = obs_dimension
        self.logger = logger
        self.device = device
        self.dataset : Dataset = dataset
        self.should_save_models = should_save_models

        self.logger.add_elements({
            "losses/train_loss": 0.0,
            "losses/validation_loss": 0.0,
            "epoch": 0,
        })
        self.logger.set_element_step_metric({
            "*": "epoch"
        })

    @abstractmethod
    def reset_models(self):
        pass

    @abstractmethod
    def train_single_epoch(self, train_loader: DataLoader):
        pass

    @abstractmethod
    def evaluate(self, validation_loader: DataLoader):
        pass

    @abstractmethod
    def save_models(self, current_epoch : int):
        pass

    @abstractmethod
    def load_model(self, path : str):
        pass

    def resolve_dataset(self, dataset: Dataset) -> list[tuple[Subset[Dataset], Subset[Dataset]]]:
        dataset_length = len(cast(Sized, dataset))

        if self.hyperparameters.validation_method_config.method == "train-validation":
            cutoff = int(dataset_length * self.hyperparameters.validation_method_config.train_size_p)
            train_idx, val_idx = np.arange(0, cutoff), np.arange(cutoff, dataset_length)
            return [(Subset(dataset, train_idx), Subset(dataset, val_idx))]
        elif self.hyperparameters.validation_method_config.method == "k-fold":
            kf_generator = KFold(n_splits=5, shuffle=True).split(X=range(dataset_length))
            return [(Subset(dataset, train_idx), Subset(dataset, val_idx)) for train_idx, val_idx in kf_generator]

        raise ValueError(f"Unsupported validation method: {self.hyperparameters.validation_method_config.method}")

    def train(self):
        datasets = self.resolve_dataset(self.dataset)

        for fold, (train_dataset, validation_dataset) in enumerate(datasets):
            self.reset_models()
            self.logger.reset("epoch")
            if len(datasets) > 1:
                self.logger.set_prefix({
                    "epoch": f"{fold}-",
                    "losses/train_loss": f"{fold}-",
                    "losses/validation_loss": f"{fold}-"
                })
            for epoch in tqdm(range(self.hyperparameters.epochs)):
                train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.hyperparameters.batch_size)
                validation_loader = DataLoader(validation_dataset, batch_size=self.hyperparameters.batch_size)
                self.train_single_epoch(train_loader)
                self.evaluate(validation_loader)

                # Logging
                self.logger.set_log_data({
                    "epoch": epoch
                })
                self.logger.log_data("losses/train_loss", "losses/validation_loss", "epoch")

                # Save checkpoint models
                if self.should_save_models:
                    self.save_models(epoch)
from abc import ABC, abstractmethod
from typing import Optional

import torch
from rl_commons.log import Logger
from rl_commons.mdp import MdpTerminationState
from torch.utils.data import Dataset


class Algorithm(ABC):

    def __init__(self, hyperparameters,
                 obs_dimension: int,
                 dataset : Dataset,
                 logger: Optional[Logger] = None,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.obs_dimension = obs_dimension
        self.logger = logger
        self.device = device
        self.dataset = dataset

    @abstractmethod
    def train(self):
        """Full implementation of training loop"""
        pass
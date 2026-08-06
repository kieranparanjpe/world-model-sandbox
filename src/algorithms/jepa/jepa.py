import copy
from typing import Optional

import torch
from rl_commons.log import Logger
from rl_commons.mdp import MdpTerminationState
from torch import nn

from src.algorithms import Algorithm
from src.datasets.dataset_sa import DatasetSA


class JEPA(Algorithm):

    def __init__(self, hyperparameters,
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

    def train(self):

        

        return
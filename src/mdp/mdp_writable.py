from abc import ABC, abstractmethod

import torch
from rl_commons.mdp import Mdp, MdpTerminationState


class MdpWritable(Mdp, ABC):

    def __init__(self, device : torch.device):
        super().__init__(device=device)

    @abstractmethod
    def set_state(self, state : torch.Tensor) -> MdpTerminationState:
        pass
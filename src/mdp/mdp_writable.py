from abc import ABC, abstractmethod

import torch
from rl_commons.mdp import Mdp


class MdpWritable(Mdp, ABC):

    def __init__(self, device : torch.Device):
        super().__init__(device=device)

    @abstractmethod
    def set_state(self, state : torch.Tensor):
        pass
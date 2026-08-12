from typing import Optional

from rl_commons.mdp import MdpGym
from torch.ao import nn

from src.mdp.mdp_gym_writable import MdpGymWritable
from src.mdp.utils import get_action_sampler


class Visualise:
    def __init__(self):
        pass

import argparse
import functools
import os
from pathlib import Path

import torch
from datetime import datetime




class Visualiser:

    def __init__(self, mdp_id : str, model : nn.Module, decoder : nn.Module, policy : Optional[nn.Module]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._mdp_main = MdpGym(mdp_id)
        self._mdp_writable = MdpGymWritable(mdp_id)
        self.model = model
        self.decoder = decoder

        self.policy = policy if policy else get_action_sampler(self._mdp_main) # need to move policy definition into
        # rl commons

    def visualise(self):
        self.algorithm.train()
        self._logger.finish()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="LunarLander-v3")
    parser.add_argument("--algorithm", "-a", help="Algorithm to use", default="jepa")
    parser.add_argument("--dataset", "-d", help="Path to dataset")
    parser.add_argument("--hyperparameters", help="Path to hyperparameter json file", default=None)
    parser.add_argument("--grid", help="Path to hyperparameter grid json file", default=None)
    parser.add_argument("--log", "-l", help="Enable log to wandb", action="store_true")
    parser.add_argument("--save", "-s", help="Enable policy saving after each update", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()





if __name__ == "__main__":
    main()

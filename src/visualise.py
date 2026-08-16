from typing import Optional

import numpy as np
from rl_commons.mdp import MdpGym, MdpConfig
from rl_commons.policies import Policy
from torch import nn

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

    def __init__(self, mdp_id : str, model : nn.Module, decoder : nn.Module,
                 policy : Optional[Policy] = None,
                 mdp_config : MdpConfig = MdpConfig(),
                 obs_rms_stats : Optional[tuple[np.ndarray, np.ndarray]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._mdp_main = MdpGym(mdp_id, device=self.device, mdp_config=mdp_config, obs_rms_stats=obs_rms_stats)
        self._mdp_writable = MdpGymWritable(mdp_id, device=self.device, mdp_config=mdp_config, obs_rms_stats=obs_rms_stats)
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
    parser.add_argument("--model", "-m", help="Path to model", required=True)
    parser.add_argument("--decoder", "-d", help="Path to decoder", required=True)
    parser.add_argument("--policy", "-p", help="Path to policy", default=None)

    return parser.parse_args()


def main():
    args = parse_args()





if __name__ == "__main__":
    main()

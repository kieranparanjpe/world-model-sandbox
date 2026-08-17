from typing import Optional

from ml_commons.stats import NormalisationStats
from rl_commons.mdp import MdpGym, MdpConfig
from rl_commons.policies import Policy
from torch import nn

from src.mdp.mdp_gym_writable import MdpGymWritable
from src.mdp.utils import get_random_policy

POLICY_OBS_NORM_KEY = "policy_obs"
WORLD_MODEL_OBS_NORM_KEY = "world_model_obs"


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
                 norm_stats : Optional[dict[str, NormalisationStats]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        norm_stats = norm_stats or {}

        self._mdp_main = MdpGym(mdp_id, device=self.device, mdp_config=mdp_config,
                                obs_rms_stats=norm_stats.get(POLICY_OBS_NORM_KEY), render_mode="human")
        self._mdp_writable = MdpGymWritable(mdp_id, device=self.device, mdp_config=mdp_config,
                                            obs_rms_stats=norm_stats.get(WORLD_MODEL_OBS_NORM_KEY), render_mode="human")

        self.model = model
        self.decoder = decoder

        self.policy = policy if policy else get_random_policy(self._mdp_main)

    def visualise(self):
        self.model.eval()
        self.decoder.eval()

        last_observation_main = self._mdp_main.reset()
        last_observation_writable = self._mdp_writable.reset()

        with torch.no_grad():
            while True:
                action_main = self.policy.forward(last_observation_main)
                action_writable = self.policy.forward(last_observation_writable)

                action_writable =

                action = self._policy.sample(action_sampler)
                next_observation, _, termination_state = self._mdp.step(action)

                self._dataset["current_observations"][timestep] = last_observation
                self._dataset["next_observations"][timestep] = next_observation
                self._dataset["actions"][timestep] = action

                if termination_state is not MdpTerminationState.IN_PROGRESS:
                    last_observation = self._mdp.reset()
                else:
                    last_observation = next_observation

        self._mdp.close()




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

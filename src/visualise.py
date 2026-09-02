from dataclasses import replace
from typing import Optional

from ml_commons.stats import NormalisationStats
from rl_commons.execution import BaseEvaluator
from rl_commons.log import BaseRecorder, NullRecorder
from rl_commons.mdp import MdpGym, MdpConfig, MdpTerminationState
from rl_commons.policies import Policy
from torch import nn

from src.algorithms.jepa.jepa_model import JEPAModel
from src.algorithms.jepa_decoder.decoder import Decoder
from src.mdp.mdp_gym_writable import MdpGymWritable
from src.mdp.utils import get_random_policy


import argparse
import functools
import os
import random
from pathlib import Path

import cv2
import torch
from datetime import datetime
import time

_WORLD_MODEL_WINDOW = "World Model"
_MAIN_WINDOW = "Main"


class Visualiser(BaseEvaluator):

    def __init__(self, task_id: str, model_path: str, decoder_path: str,
                 policy_id_path: Optional[tuple[str, str]] = None, mdp_config: MdpConfig = MdpConfig(),
                 sync_reset: bool = False,
                 sync_timesteps : int = 10**10,
                 render_mode: str = "rgb_array"):
        self._mdp_config = replace(mdp_config, normalise_obs=False, normalise_reward=False)
        super().__init__(task_id, mdp_config=self._mdp_config, render_mode=render_mode)

        self.sync_reset = sync_reset
        self._sync_timesteps = sync_timesteps

        self._mdp_world_model = MdpGymWritable(task_id, device=self.device, mdp_config=self._mdp_config, render_mode="rgb_array")

        self.model = JEPAModel.load(model_path, map_location=self.device)
        self.decoder = Decoder.load(decoder_path, map_location=self.device)

        self.policy = Policy.load(policy_id_path[1],
                                  map_location=self.device,
                                  obs_dimension=self._mdp.obs_dimension,
                                  action_dimension=self._mdp.action_dimension,
                                  policy_id=policy_id_path[0]) \
            if policy_id_path else get_random_policy(self._mdp)

    def standardize(self, value: torch.Tensor, norm_stats: NormalisationStats) -> torch.Tensor:
        return (value - norm_stats.mean_t(device=self.device)) / norm_stats.std_t(device=self.device)

    def unstandardize(self, value: torch.Tensor, norm_stats: NormalisationStats):
        return (value * norm_stats.std_t(device=self.device)) + norm_stats.mean_t(device=self.device)

    def _run(self):
        self.model.eval()
        self.decoder.eval()

        seed = random.randint(0, 2**31 - 1)
        last_observation_main = self._mdp.reset(seed=seed)
        self._mdp_world_model.reset(seed=seed)
        self._mdp_world_model.set_state(last_observation_main)
        last_observation_writable = last_observation_main

        last_observation_writable_world_model_norm = self.standardize(last_observation_writable, self.model.obs_norm_stats)
        last_observation_writable_latent = self.model.encoder(last_observation_writable_world_model_norm)

        timestep_main, timestep_world_model = 0, 0

        with torch.no_grad():
            while not self._stop.is_set():
                if self.sync_reset and timestep_main % self._sync_timesteps == 0:
                    self._mdp_world_model.set_state(last_observation_main)
                    last_observation_writable = last_observation_main
                    last_observation_writable_world_model_norm = self.standardize(last_observation_writable,
                                                                                  self.model.obs_norm_stats)
                    last_observation_writable_latent = self.model.encoder(last_observation_writable_world_model_norm)

                # MAIN
                last_obs_policy_norm = self.standardize(last_observation_main, self.policy.obs_norm_stats)

                action = self.policy.sample_action(self.policy.forward(last_obs_policy_norm))

                next_obs_main, _, termination_state_main = self._mdp.step(action)

                frame_main = cv2.cvtColor(self._mdp.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(frame_main, f"t={timestep_main}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(_MAIN_WINDOW, frame_main)

                # WORLD MODEL — reuse same action from main
                if self._mdp_world_model.discrete:
                    action = torch.nn.functional.one_hot(action, self._mdp_world_model.action_dimension)

                action_world_norm = self.standardize(action, self.model.action_norm_stats)

                next_obs_latent = self.model.predictor(last_observation_writable_latent, action_world_norm)

                next_obs_decoder_norm = self.decoder(next_obs_latent)
                next_obs_world_model = self.unstandardize(next_obs_decoder_norm, self.decoder.obs_norm_stats)

                termination_state_world_model = self._mdp_world_model.set_state(next_obs_world_model)

                frame = cv2.cvtColor(self._mdp_world_model.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"t={timestep_world_model}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(_WORLD_MODEL_WINDOW, frame)
                cv2.waitKey(1)

                # Resetting:
                if self.sync_reset:
                    termination_state_world_model = termination_state_main

                reset_seed = None
                if self.sync_reset and termination_state_main is not MdpTerminationState.IN_PROGRESS:
                    reset_seed = random.randint(0, 2**31 - 1)

                if termination_state_main is not MdpTerminationState.IN_PROGRESS:
                    last_observation_main = self._mdp.reset(seed=reset_seed)
                    timestep_main = 0

                else:
                    last_observation_main =  next_obs_main
                    timestep_main += 1

                if termination_state_world_model is not MdpTerminationState.IN_PROGRESS:
                    last_observation_writable = self._mdp_world_model.reset(seed=reset_seed)
                    if reset_seed is not None:
                        # Same seed as main -> identical hidden reset-time randomness (terrain,
                        # initial impulse). Pin the observed dims too, for exact parity.
                        self._mdp_world_model.set_state(last_observation_main)
                        last_observation_writable = last_observation_main
                    last_observation_writable_world_model_norm = self.standardize(last_observation_writable,
                                                                                  self.model.obs_norm_stats)
                    last_observation_writable_latent = self.model.encoder(last_observation_writable_world_model_norm)

                    timestep_world_model = 0
                else:
                    last_observation_writable, last_observation_writable_latent = next_obs_world_model, next_obs_latent
                    timestep_world_model += 1

                time.sleep(0.1)

        self._mdp_world_model.close()
        cv2.destroyAllWindows()




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="LunarLander-v3")
    parser.add_argument("--model", "-m", help="Path to model", required=True)
    parser.add_argument("--decoder", "-d", help="Path to decoder", required=True)
    parser.add_argument("--weights", "-w", help="Path to policy", default=None)
    parser.add_argument("--policy", "-p", help="Policy type", default=None)


    parser.add_argument("--sync", help="Should we sync mdp resetting", action="store_true")
    parser.add_argument("--sync_timesteps", "-st", help="Number of timesteps before syncing", default=10**10)


    return parser.parse_args()


def main():
    args = parse_args()
    policy_id_path = (args.policy, args.weights) if args.policy and args.weights else None
    visualiser = Visualiser(args.environment, args.model, args.decoder, policy_id_path, sync_reset=args.sync,
                            sync_timesteps=int(args.sync_timesteps))
    visualiser.evaluate()


if __name__ == "__main__":
    main()

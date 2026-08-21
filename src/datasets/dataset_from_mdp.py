import argparse
from datetime import datetime
from typing import Optional

import torch.distributions
from rl_commons.mdp import Mdp, MdpTerminationState, MdpGym, MdpConfig
from rl_commons.policies import Policy
from tqdm.auto import tqdm
from pathlib import Path
import torch

from src.mdp.utils import get_random_policy


class DatasetFromMDP:

    def __init__(self, mdp : Mdp, timesteps : int, mdp_id : str, policy_id_path: Optional[tuple[str, str]] = None):
        self._mdp = mdp
        self._timesteps = timesteps
        time = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
        self._output_path = Path(__file__).resolve().parents[2] / "datasets" / mdp_id / "state-action" / (f"dataset"
                                                                                                      f"_{time}.pt")
        self._policy = Policy.load(policy_id_path[1],
                                  map_location=self._mdp.device,
                                  obs_dimension=self._mdp.obs_dimension,
                                  action_dimension=self._mdp.action_dimension,
                                  policy_id=policy_id_path[0]) \
            if policy_id_path else get_random_policy(self._mdp)
        self._policy = get_random_policy(self._mdp)

        self._dataset = {
            "current_observations" : torch.zeros((timesteps, self._mdp.obs_dimension), dtype=torch.float32,
                                                 device=self._mdp.device),
            "next_observations": torch.zeros((timesteps, self._mdp.obs_dimension), dtype=torch.float32,
                                                 device=self._mdp.device),
            "actions": torch.zeros((timesteps, self._mdp.action_dimension), dtype=torch.float32,
                                                 device=self._mdp.device),
            "episode": torch.zeros((timesteps, 1), dtype=torch.int64,
                                   device=self._mdp.device)
        }

    def collect(self):
        last_observation = self._mdp.reset()

        action_sampler = self._policy.forward(last_observation)

        episode = torch.tensor(0, device=self._mdp.device)

        with torch.no_grad():
            for timestep in tqdm(range(self._timesteps)):
                action = self._policy.sample_action(action_sampler)
                next_observation, _, termination_state = self._mdp.step(action)

                if self._mdp.discrete:
                    action = torch.nn.functional.one_hot(action, self._mdp.action_dimension)

                self._dataset["current_observations"][timestep] = last_observation
                self._dataset["next_observations"][timestep] = next_observation
                self._dataset["actions"][timestep] = action
                self._dataset["episode"][timestep] = episode

                if termination_state is not MdpTerminationState.IN_PROGRESS:
                    episode += 1
                    last_observation = self._mdp.reset()

                else:
                    last_observation = next_observation

        self._mdp.close()

    def save(self):
        file_path = Path(self._output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._dataset, file_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--timesteps", "-t", help="Number of samples to collect", default=10000)
    parser.add_argument("--weights", "-w", help="Path to policy", default=None)
    parser.add_argument("--policy", "-p", help="Policy type", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdp = MdpGym(args.environment, device, mdp_config=MdpConfig(normalise_obs=False, normalise_reward=False))

    policy_id_path = (args.policy, args.weights) if args.policy and args.weights else None


    dataset_from_mdp = DatasetFromMDP(mdp, int(args.timesteps), args.environment, policy_id_path=policy_id_path)
    dataset_from_mdp.collect()
    dataset_from_mdp.save()


if __name__ == "__main__":
    main()
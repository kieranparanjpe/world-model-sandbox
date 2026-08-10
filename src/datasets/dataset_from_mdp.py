import argparse
from datetime import datetime

import torch.distributions
from rl_commons.mdp import Mdp, MdpTerminationState, MdpGym
from tqdm.auto import tqdm
from pathlib import Path
import torch

class DatasetFromMDP:

    def __init__(self, mdp : Mdp, timesteps : int, mdp_id : str):
        self._mdp = mdp
        self._timesteps = timesteps
        time = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
        self._output_path = Path(__file__).resolve().parents[2] / "datasets" / mdp_id / f"dataset_{time}.pt"

        self._action_sampler = self.init_action_sampler()

        self._dataset = {
            "current_observations" : torch.zeros((timesteps, self._mdp.obs_dimension), dtype=torch.float32,
                                                 device=torch.device("cpu")),
            "next_observations": torch.zeros((timesteps, self._mdp.obs_dimension), dtype=torch.float32,
                                                 device=torch.device("cpu")),
            "actions": torch.zeros((timesteps, self._mdp.action_dimension), dtype=torch.float32,
                                                 device=torch.device("cpu"))
        }

    def init_action_sampler(self):
        if self._mdp.discrete:
            return torch.distributions.Categorical(logits=torch.ones(self._mdp.action_dimension))
        else:
            return torch.distributions.Uniform(low=self._mdp.action_range[:, 0], high=self._mdp.action_range[:, 1])

    def collect(self):
        last_observation = self._mdp.reset()

        for timestep in tqdm(range(self._timesteps)):
            action = self._action_sampler.sample()
            next_observation, _, termination_state = self._mdp.step(action)

            self._dataset["current_observations"][timestep] = last_observation
            self._dataset["next_observations"][timestep] = next_observation
            self._dataset["actions"][timestep] = action

            if termination_state is not MdpTerminationState.IN_PROGRESS:
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

    return parser.parse_args()


def main():
    args = parse_args()

    mdp = MdpGym(args.environment, torch.device("cpu"))

    dataset_from_mdp = DatasetFromMDP(mdp, int(args.timesteps), args.environment)
    dataset_from_mdp.collect()
    dataset_from_mdp.save()


if __name__ == "__main__":
    main()
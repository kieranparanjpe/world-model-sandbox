import argparse
import functools
import os

import torch
from tqdm.auto import tqdm
from datetime import datetime

from rl_commons.config import RunInfo
from rl_commons.execution import gridsearch, BaseTrainer, run_one
from rl_commons.mdp import MdpTerminationState
from src.config import RunConfig, load_config, load_grid_configs
from src.algorithms import PPO
from src.algorithms.policies import PolicyFactory


class Trainer(BaseTrainer):

    def __init__(self, run_info: RunInfo, run_config: RunConfig,
                 logging=True, save_policy=False, record=False):
        super().__init__(
            run_info=run_info,
            run_config=run_config,
            mdp_config=run_config.mdp,
            entity="kieranparanjpe-mcgill-university",
            project="RL_Project1",
            log_elements={
                "charts/episodic_return": 0.0,
                "charts/episode_length": 0,
                "global_step": 0,
            },
            logging=logging,
            record=record,
            total_timesteps=run_config.algorithm.n_timesteps,
        )

        self._should_save_policy = save_policy
        if self._should_save_policy:
            self._create_policy_folder()

        self.policy = PolicyFactory.build_policy(
            self._run_info.policy_id,
            self._mdp.obs_dimension,
            self._mdp.action_dimension,
            run_config.policy,
        ).to(self.device)

        if self._run_info.algorithm_id == 'ppo':
            self.algorithm = PPO(
                run_config.algorithm, self.policy,
                self._mdp.obs_dimension, self._mdp.action_dimension, self._mdp.discrete,
                logger=self._logger, device=self.device,
                value_fn_config=run_config.value_fn,
            )

    def _create_policy_folder(self):
        directory_path = self._run_info.local_folder_path("saved_policies")
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def _save_policy(self, timestep):
        n_timesteps = self._run_config.algorithm.n_timesteps
        width = len(str(n_timesteps))
        save_dict = {"policy": self.policy.state_dict()}
        if (stats := self._mdp.obs_rms_stats) is not None:
            save_dict["norm_stats"] = {
                "obs_mean": torch.tensor(stats[0]),
                "obs_var": torch.tensor(stats[1]),
            }
        torch.save(save_dict, f'{self._run_info.local_folder_path("saved_policies")}/policy_{timestep:0{width}d}.pth')

    def run(self):
        last_observation = self._mdp.reset()
        episode_number = 0
        n_timesteps = self._run_config.algorithm.n_timesteps
        for timestep in tqdm(range(n_timesteps)):
            action, log_prob_action = self.algorithm.sample_action(last_observation)

            next_observation, reward, termination_state = self._mdp.step(action)

            updated_policy = self.algorithm.update_and_observe(last_observation, next_observation, action,
                                                               log_prob_action, reward,
                                                               termination_state, timestep)

            if ((updated_policy and self._should_save_policy and episode_number % 500 == 0) or
                    timestep == n_timesteps - 1):
                self._save_policy(timestep)

            self._logger.sum_log_data({
                "charts/episodic_return": reward,
                "charts/episode_length": 1,
            })
            if termination_state is not MdpTerminationState.IN_PROGRESS:
                last_observation = self._mdp.reset()

                self._logger.set_log_data({"global_step": timestep})
                self._logger.log_data("charts/episodic_return", "charts/episode_length", "global_step")
                self._logger.reset("charts/episodic_return", "charts/episode_length")

                self._recorder.new_episode = True
                episode_number += 1

            else:
                last_observation = next_observation

                self._recorder.new_episode = False

        self._mdp.close()
        self._logger.upload_videos(self._recorder)
        self._logger.finish()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="CartPole-v1")
    parser.add_argument("--algorithm", "-a", help="Algorithm to use", default="ppo")
    parser.add_argument("--policy", "-p", help="Policy Id to use", default="categorical")
    parser.add_argument("--hyperparameters", help="Path to hyperparameter json file", default=None)
    parser.add_argument("--grid", help="Path to hyperparameter grid json file", default=None)

    parser.add_argument("--log", "-l", help="Enable log to wandb", action="store_true")
    parser.add_argument("--save", "-s", help="Enable policy saving after each update", action="store_true")
    parser.add_argument("--record", "-r", help="Enable episode recording", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    now = datetime.now()

    factory = functools.partial(Trainer, logging=args.log, save_policy=args.save, record=args.record)
    _run_one = functools.partial(run_one, args=args, now=now, trainer_factory=factory)

    if args.grid is not None:
        configs = load_grid_configs(args.grid, args.algorithm, args.policy)
        gridsearch(_run_one, configs)
    elif args.hyperparameters is not None:
        _run_one(load_config(args.hyperparameters, args.algorithm, args.policy), None)
    else:
        _run_one(RunConfig(), None)


if __name__ == "__main__":
    main()

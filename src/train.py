import argparse
import functools
import os
from pathlib import Path
from typing import Optional

import torch
from datetime import datetime

from ml_commons.config import RunInfo
from ml_commons.execution import gridsearch, run_one
from ml_commons.networks import SaveableNetwork
from rl_commons.execution import BaseTrainer as BaseTrainerRL
from rl_commons.mdp import MdpConfig

from src.algorithms import Algorithm
from src.algorithms.algorithm_factory import create_algorithm
from src.config import RunConfig, load_config, load_grid_configs, RunInfoSupervised
from src.datasets.dataset_factory import create_dataset
from src.datasets.norm_stats_keys import OBS_NORM_KEY

from src.datasets.dataset_sa import DatasetSA


class Trainer(BaseTrainerRL):

    def __init__(self, run_info: RunInfoSupervised, run_config: RunConfig, dataset_path : str, dataset_type : str,
                 logging=True, save_policy=False,
                 normalise_obs=False, obs_norm_source : Optional[str] = None):
        super().__init__(
            run_info=run_info,
            run_config=run_config,
            mdp_config=run_config.mdp,
            entity="kieranparanjpe-mcgill-university",
            project="World-Model-Sandbox",
            log_elements={},
            logging=logging
        )

        self._dataset = create_dataset(dataset_type, path=Path(dataset_path))

        if obs_norm_source is not None:
            source_stats = SaveableNetwork.load_norm_stats(obs_norm_source, map_location=self.device)
            self._dataset.apply_obs_normalization(source_stats[OBS_NORM_KEY])
        elif normalise_obs:
            self._dataset.apply_obs_normalization(self._dataset.get_norm_stats()[OBS_NORM_KEY])

        self.algorithm : Algorithm = create_algorithm(self._run_info.algorithm_id,
                                          hyperparameters=self._run_config.algorithm,
                                          run_info=self._run_info,
                                          obs_dimension=self._mdp.obs_dimension,
                                          action_dimension=self._mdp.action_dimension,
                                          dataset=self._dataset,
                                          logger=self._logger,
                                          device=self.device,
                                          should_save_models=save_policy)


    def run(self):
        self.algorithm.train()
        self._logger.finish()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="LunarLander-v3")
    parser.add_argument("--algorithm", "-a", help="Algorithm to use", default="jepa")
    parser.add_argument("--dataset", "-d", help="Path to dataset")
    parser.add_argument("--hyperparameters", help="Path to hyperparameter json file", default=None)
    parser.add_argument("--grid", help="Path to hyperparameter grid json file", default=None)
    parser.add_argument("--dataset_type", "-dt", help="dataset type", default="sa")
    parser.add_argument("--log", "-l", help="Enable log to wandb", action="store_true")
    parser.add_argument("--save", "-s", help="Enable policy saving after each update", action="store_true")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalise-obs", help="Self-compute obs norm stats from this dataset and apply before training",
                            action="store_true")
    norm_group.add_argument("--obs-norm-source",
                            help="Path to a checkpoint (e.g. a trained jepa model) to load obs norm stats from and apply to this dataset",
                            default=None)

    return parser.parse_args()


def make_trainer(run_config, index, args, now):
    run_info = RunInfoSupervised(
        task_id=args.environment,
        dataset=args.dataset,
        algorithm_id=args.algorithm,
        grid_index=index,
        time=now,
    )
    return Trainer(run_info, run_config, dataset_path=args.dataset, logging=args.log, save_policy=args.save,
                   dataset_type=args.dataset_type, normalise_obs=args.normalise_obs,
                   obs_norm_source=args.obs_norm_source)

def main():
    args = parse_args()
    now = datetime.now()

    _run_one = functools.partial(
        run_one,
        trainer_factory=functools.partial(make_trainer, args=args, now=now)
    )

    if args.grid is not None:
        configs = load_grid_configs(args.grid, args.algorithm)
        gridsearch(_run_one, configs)
    elif args.hyperparameters is not None:
        _run_one(load_config(args.hyperparameters, args.algorithm), None)
    else:
        _run_one(RunConfig(), None)


if __name__ == "__main__":
    main()

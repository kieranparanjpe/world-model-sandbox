import argparse
from pathlib import Path
from typing import Optional

import torch
from datetime import datetime

from ml_commons.config import RunInfo
from ml_commons.networks import SaveableNetwork
from rl_commons.execution import BaseEvaluator
from torch.utils.data import DataLoader

from src.log.console_logger import ConsoleLogger
from src.algorithms import Algorithm
from src.algorithms.algorithm_factory import create_algorithm
from src.config import RunConfig, load_config, RunInfoSupervised
from src.datasets.dataset_factory import create_dataset

from src.datasets.dataset_sa import DatasetSA


class Evaluator(BaseEvaluator):

    def __init__(self, run_info: RunInfo, run_config: RunConfig, dataset_path: str, dataset_type: str, task_id: str,
                 model_path: str, normalise_obs=False, obs_norm_source: Optional[str] = None, normalise_action=False):
        super().__init__(run_info.task_id, run_config.mdp)
        self._run_info = run_info
        self._run_config = run_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dataset = create_dataset(dataset_type, path=Path(dataset_path))

        if obs_norm_source is not None:
            # keyed by the source model's attribute name (Policy/JEPAModel/Decoder all use "obs_norm_stats")
            source_stats = SaveableNetwork.load_norm_stats(obs_norm_source, map_location=self.device)
            self._dataset.apply_obs_normalization(source_stats["obs_norm_stats"])
        elif normalise_obs:
            self._dataset.normalise_obs()

        if normalise_action:
            self._dataset.normalise_actions()

        logger = ConsoleLogger(self._run_info, "", "", vars(self._run_info), {})

        self.algorithm : Algorithm = create_algorithm(self._run_info.algorithm_id,
                                          hyperparameters=self._run_config.algorithm,
                                          run_info=self._run_info,
                                          obs_dimension=self._mdp.obs_dimension,
                                          action_dimension=self._mdp.action_dimension,
                                          dataset=self._dataset,
                                          logger=logger,
                                          device=self.device,
                                          should_save_models=False)

        self.algorithm.load_model(model_path)

    def _run(self):
        validation_loader = DataLoader(self._dataset, batch_size=self._run_config.algorithm.batch_size)
        self.algorithm.evaluate(validation_loader)
        self.algorithm.logger.log_data("losses/validation_loss")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", "-e", help="Environment Id to run", default="LunarLander-v3")
    parser.add_argument("--algorithm", "-a", help="Algorithm to use", default="jepa")
    parser.add_argument("--dataset", "-d", help="Path to dataset")
    parser.add_argument("--model", "-m", help="Path to saved model checkpoint", required=True)
    parser.add_argument("--hyperparameters", help="Path to hyperparameter json file", default=None)
    parser.add_argument("--dataset_type", "-dt", help="dataset type", default="sa")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalise-obs", help="Self-compute obs norm stats from this dataset and apply before training",
                            action="store_true")
    norm_group.add_argument("--obs-norm-source",
                            help="Path to a checkpoint (e.g. a trained jepa model) to load obs norm stats from and apply to this dataset",
                            default=None)
    parser.add_argument("--normalise-action", help="Self-compute action norm stats from this dataset and apply before training",
                        action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    now = datetime.now()

    run_config = load_config(args.hyperparameters, args.algorithm) if args.hyperparameters is not None else RunConfig()

    run_info = RunInfoSupervised(
        task_id=args.environment,
        dataset=args.dataset,
        algorithm_id=args.algorithm,
        grid_index=None,
        time=now,
    )

    evaluator = Evaluator(run_info, run_config, dataset_path=args.dataset, dataset_type=args.dataset_type,
                          task_id=args.environment, model_path=args.model, normalise_obs=args.normalise_obs,
                          obs_norm_source=args.obs_norm_source, normalise_action=args.normalise_action)
    evaluator.evaluate()


if __name__ == "__main__":
    main()

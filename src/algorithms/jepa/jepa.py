import copy
import os
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from ml_commons.config import RunInfo
from ml_commons.log import Logger
from torch import nn, optim
from torch.utils.data import DataLoader

from src.algorithms import Algorithm
from src.algorithms.jepa.encoder import Encoder
from src.algorithms.jepa.jepa_config import JepaConfig
from src.algorithms.jepa.jepa_model import JEPAModel
from src.algorithms.jepa.predictor import Predictor
from src.algorithms.utils import save_model
from src.datasets.dataset_sa import DatasetSA
from src.datasets.norm_stats_keys import OBS_NORM_KEY, ACTION_NORM_KEY

def jepa_factory(hyperparameters : JepaConfig,
                 run_info : RunInfo,
                 obs_dimension : int,
                 action_dimension : int,
                 dataset : DatasetSA,
                 logger : Optional[Logger],
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
    encoder_factory = lambda : Encoder(obs_dimension, hyperparameters.encoder_config)
    predictor_factory = lambda : Predictor(action_dimension, hyperparameters.predictor_config)

    return JEPA(hyperparameters, run_info, obs_dimension, encoder_factory, predictor_factory, dataset, logger, device,
                should_save_models)


class JEPA(Algorithm):
    model: JEPAModel
    label_encoder: Encoder

    encoder_optimiser: torch.optim.Optimizer
    predictor_optimiser: torch.optim.Optimizer

    hyperparameters: JepaConfig
    dataset: DatasetSA

    def __init__(self, hyperparameters : JepaConfig,
                 run_info : RunInfo,
                 obs_dimension: int,
                 encoder_factory : Callable[[], Encoder],
                 predictor_factory : Callable[[], Predictor],
                 dataset : DatasetSA,
                 logger: Optional[Logger] = None,
                 device: torch.device = torch.device('cpu'),
                 should_save_models=False):
        super().__init__(hyperparameters, run_info, obs_dimension, dataset, logger, device, should_save_models)

        self.dataset.set_number_steps(hyperparameters.lookahead_steps)

        self.encoder_factory = encoder_factory
        self.predictor_factory = predictor_factory
        self.reset_models()

        self.criterion = nn.MSELoss(reduction='none')

    # noinspection PyAttributeOutsideInit
    def reset_models(self):
        self.model = JEPAModel(self.encoder_factory(), self.predictor_factory())
        self.label_encoder = copy.deepcopy(self.model.encoder)

        self.model.to(self.device)
        self.label_encoder.to(self.device)

        self.encoder_optimiser = optim.Adam(self.model.encoder.parameters(),
                                            lr=self.hyperparameters.encoder_lr,
                                            weight_decay=self.hyperparameters.encoder_regularization)
        self.predictor_optimiser = optim.Adam(self.model.predictor.parameters(),
                                              lr=self.hyperparameters.predictor_lr,
                                              weight_decay=self.hyperparameters.predictor_regularization)

        # Freeze grads for label encoder. it is updated with EMA
        for param in self.label_encoder.parameters():
            param.requires_grad = False

    def save_models(self, current_epoch : int):
        model_path = self.run_info.local_folder_path("saved_networks/jepa/model")
        os.makedirs(model_path, exist_ok=True)

        width = len(str(self.hyperparameters.epochs))

        norm_stats = self.dataset.get_norm_stats()
        self.model.obs_norm_stats = norm_stats[OBS_NORM_KEY]
        self.model.action_norm_stats = norm_stats[ACTION_NORM_KEY]
        self.model.save(f'{model_path}/model_{current_epoch:0{width}d}.pt')

    def load_model(self, path : str):
        self.model = JEPAModel.load(path, map_location=self.device)
        self.label_encoder = copy.deepcopy(self.model.encoder)
        for param in self.label_encoder.parameters():
            param.requires_grad = False

    def _get_curriculum_k(self, epoch: int) -> int:
        if self.hyperparameters.curriculum_epochs > 0:
            progress = min(1.0, epoch / self.hyperparameters.curriculum_epochs)
            return 1 + int((self.dataset.number_steps - 1) * progress)
        return self.dataset.number_steps

    def _autoregressive_rollout(self, current_observations, actions, next_observations, max_k):
        step_losses = []
        raw_losses = []
        preds_norm_list = []
        targets_norm_list = []

        # k=1 rollout
        pred_enc, _ = self.model(current_observations[:, 0], actions[:, 0])
        target_enc = self.label_encoder(next_observations[:, 0]).detach()

        pred_norm = F.normalize(pred_enc, p=2, dim=-1)
        target_norm = F.normalize(target_enc, p=2, dim=-1)

        step_losses.append(self.criterion(pred_norm, target_norm).mean(dim=-1))
        raw_losses.append(self.criterion(pred_enc, target_enc).mean(dim=-1))
        preds_norm_list.append(pred_norm)
        targets_norm_list.append(target_norm)

        # k>1 autoregressive loop
        for i in range(1, max_k):
            pred_enc = self.model.predictor(pred_enc, actions[:, i])
            target_enc = self.label_encoder(next_observations[:, i]).detach()

            # L2 Normalize before loss to prevent explosion
            pred_norm = F.normalize(pred_enc, p=2, dim=-1)
            target_norm = F.normalize(target_enc, p=2, dim=-1)

            step_losses.append(self.criterion(pred_norm, target_norm).mean(dim=-1))
            raw_losses.append(self.criterion(pred_enc, target_enc).mean(dim=-1))
            preds_norm_list.append(pred_norm)
            targets_norm_list.append(target_norm)

        return step_losses, raw_losses, preds_norm_list, targets_norm_list

    def _calculate_discounted_loss(self, step_losses, mask, max_k):
        gamma = self.hyperparameters.lookahead_gamma
        mask_k = mask[:, :max_k].t()  # [max_k, B]
        gammas = torch.tensor([gamma ** i for i in range(max_k)], device=self.device).view(-1, 1)

        stacked_losses = torch.stack(step_losses)  # [max_k, B]
        discounted_weights = mask_k * gammas  # [max_k, B]

        per_sample_loss = (stacked_losses * discounted_weights).sum(dim=0)  # [B]
        per_sample_weights = discounted_weights.sum(dim=0).clamp(min=1e-8)  # [B]

        return (per_sample_loss / per_sample_weights).mean()

    def _compute_metrics(self, step_losses, raw_losses, preds, targets, mask):
        max_k = len(step_losses)
        mask_k = mask[:, :max_k].t()  # [max_k, B]

        # 1. Standard Deviations (to detect representation collapse)
        all_preds = torch.stack(preds).view(-1, preds[0].size(-1))
        all_targets = torch.stack(targets).view(-1, targets[0].size(-1))

        pred_std = all_preds.std(dim=0).mean().item() if all_preds.size(0) > 1 else 0.0
        target_std = all_targets.std(dim=0).mean().item() if all_targets.size(0) > 1 else 0.0

        # 2. Raw & Undiscounted Losses
        stacked_step = torch.stack(step_losses) * mask_k
        stacked_raw = torch.stack(raw_losses) * mask_k
        valid_count = mask_k.sum().clamp(min=1e-8)

        loss_undiscounted = (stacked_step.sum() / valid_count).item()
        loss_raw = (stacked_raw.sum() / valid_count).item()

        metrics = {
            "encodings_std": target_std,
            "predictions_std": pred_std,
            "loss_undiscounted": loss_undiscounted,
            "loss_raw": loss_raw,
        }

        # 3. Horizon Degradation Metrics
        step_avgs = []
        for i in range(max_k):
            valid_b = mask_k[i].sum().clamp(min=1e-8)
            step_avg = (stacked_step[i].sum() / valid_b).item()
            step_avgs.append(step_avg)
            metrics[f"step_loss_{i+1:02d}"] = step_avg

        if len(step_avgs) > 1 and step_avgs[0] > 0:
            metrics["degradation_ratio"] = step_avgs[-1] / step_avgs[0]
        else:
            metrics["degradation_ratio"] = 1.0

        return metrics

    def _get_metric_keys(self, max_k: int, prefix: str) -> dict[str, float]:
        keys = {
            f"metrics/{prefix}_encodings_std": 0.0,
            f"metrics/{prefix}_predictions_std": 0.0,
            f"metrics/{prefix}_loss_undiscounted": 0.0,
            f"metrics/{prefix}_loss_raw": 0.0,
            f"metrics/{prefix}_degradation_ratio": 0.0,
        }
        for i in range(1, max_k + 1):
            keys[f"metrics/{prefix}_step_loss_{i:02d}"] = 0.0
        return keys

    def loss(self, current_observations, actions, next_observations, mask, epoch: int):
        mask = mask.float()
        max_k = self._get_curriculum_k(epoch)

        step_losses, raw_losses, preds, targets = self._autoregressive_rollout(
            current_observations, actions, next_observations, max_k
        )

        final_loss = self._calculate_discounted_loss(step_losses, mask, max_k)
        metrics_dict = self._compute_metrics(step_losses, raw_losses, preds, targets, mask)

        return final_loss, metrics_dict

    def extract_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (batch["current_observations"].to(self.device),
                batch["actions"].to(self.device),
                batch["next_observations"].to(self.device),
                batch["valid"].to(self.device))

    def train_single_epoch(self, train_loader : DataLoader[DatasetSA], epoch: int):
        self.model.train()
        self.label_encoder.train()

        max_k = self._get_curriculum_k(epoch)
        metric_keys = self._get_metric_keys(max_k, "train")
        self.logger.add_elements(metric_keys)
        self.logger.reset("losses/train_loss", *metric_keys.keys())

        for batch in train_loader:
            current_observations, actions, next_observations, mask = self.extract_batch(batch)

            loss, metrics = self.loss(current_observations, actions, next_observations, mask, epoch)

            self.model.zero_grad()
            loss.backward()

            self.encoder_optimiser.step()
            self.predictor_optimiser.step()

            with torch.no_grad():
                # Update label encoder weights with EMA
                for target_param, online_param in zip(self.label_encoder.parameters(),
                                                       self.model.encoder.parameters()):
                    target_param.lerp_(online_param, 1.0 - self.hyperparameters.label_encoder_ema_momentum)

                # Logging and stats
                batch_length = current_observations.size(0)
                total_samples = len(train_loader.dataset) # type: ignore
                weight = batch_length / total_samples

                self.logger.sum_log_data({"losses/train_loss": loss.item() * weight})
                self.logger.sum_log_data({f"metrics/train_{k}": v * weight for k, v in metrics.items()})

    def evaluate(self, validation_loader : DataLoader[DatasetSA], epoch: int):
        self.model.eval()
        self.label_encoder.eval()

        max_k = self._get_curriculum_k(epoch)
        metric_keys = self._get_metric_keys(max_k, "validation")
        self.logger.add_elements(metric_keys)
        self.logger.reset("losses/validation_loss", *metric_keys.keys())

        with torch.no_grad():
            for batch in validation_loader:
                current_observations, actions, next_observations, mask = self.extract_batch(batch)

                loss, metrics = self.loss(current_observations, actions, next_observations, mask, epoch)

                batch_length = current_observations.size(0)
                total_samples = len(validation_loader.dataset)  # type: ignore
                weight = batch_length / total_samples

                self.logger.sum_log_data({"losses/validation_loss": loss.item() * weight})
                self.logger.sum_log_data({f"metrics/validation_{k}": v * weight for k, v in metrics.items()})

from abc import ABC
from dataclasses import dataclass

from src.algorithms.validation_method_config import ValidationMethodConfig, TrainValidationConfig


@dataclass
class AlgorithmConfig:
    epochs: int = 1_000_000
    validation_method_config: ValidationMethodConfig = TrainValidationConfig()


@dataclass
class JepaConfig(AlgorithmConfig):
    encoder_lr : float = 0.01
    predictor_lr : float = 0.01
    label_predictor_gamma : float = 0.99
    encoder_regularization : float = 0
    predictor_regularization : float = 0


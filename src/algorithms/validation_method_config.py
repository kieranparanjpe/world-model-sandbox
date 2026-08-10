from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class ValidationMethodConfig(ABC):
    method : Literal["none"] = "none"

@dataclass
class TrainValidationConfig(ValidationMethodConfig):
    method : Literal["train-validation"] = "train-validation"
    train_size_p : float = 0.8

@dataclass
class KFoldValidationConfig(ValidationMethodConfig):
    method : Literal["k-fold"] = "k-fold"
    number_folds : int = 5
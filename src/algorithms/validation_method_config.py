from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ValidationMethodConfig(ABC):
    @abstractmethod
    @property
    def method(self):
        return ""

@dataclass
class TrainValidationConfig(ValidationMethodConfig):
    @property
    def method(self):
        return "train-validation"

    train_size_p : float = 0.8

@dataclass
class KFoldValidationConfig(ValidationMethodConfig):
    @property
    def method(self):
        return "k-fold"

    number_folds : int = 5
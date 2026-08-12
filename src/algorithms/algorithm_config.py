from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated, Union
from pydantic import Field

from src.algorithms.validation_method_config import ValidationMethodConfig, TrainValidationConfig, KFoldValidationConfig


AnyValidationConfig = Annotated[
    Union[TrainValidationConfig, KFoldValidationConfig, ValidationMethodConfig],
    Field(discriminator='method')
]

@dataclass
class AlgorithmConfig:
    epochs: int = 1_000_000
    batch_size: int = 64
    validation_method_config: AnyValidationConfig = field(default_factory=TrainValidationConfig)

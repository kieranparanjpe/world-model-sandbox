from dataclasses import dataclass, field

from src.algorithms.algorithm_config import AlgorithmConfig
from src.algorithms.jepa.encoder_config import EncoderConfig
from src.algorithms.jepa.predictor_config import PredictorConfig


@dataclass
class JepaConfig(AlgorithmConfig):
    encoder_lr : float = 0.01
    predictor_lr : float = 0.01
    label_encoder_ema_momentum : float = 0.99
    encoder_regularization : float = 0
    predictor_regularization : float = 0

    encoder_config : EncoderConfig = field(default_factory=EncoderConfig)
    predictor_config : PredictorConfig = field(default_factory=PredictorConfig)
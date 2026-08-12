from dataclasses import dataclass, field

from src.algorithms.algorithm_config import AlgorithmConfig
from src.algorithms.jepa.encoder_config import EncoderConfig
from src.algorithms.jepa.predictor_config import PredictorConfig
from src.algorithms.jepa_decoder.decoder_config import DecoderConfig


@dataclass
class JepaDecoderConfig(AlgorithmConfig):
    decoder_lr : float = 0.01
    decoder_regularization : float = 0

    decoder_config : DecoderConfig = field(default_factory=DecoderConfig)

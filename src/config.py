from __future__ import annotations

from dataclasses import dataclass, field

from ml_commons.config import ConfigLoader, RunInfo

# Re-export so callers that import RunInfo/ConfigLoader from src.config continue to work.
__all__ = ["RunConfig", "load_config", "load_grid_configs"]

from rl_commons.mdp import MdpConfig

from src.algorithms.algorithm_config import AlgorithmConfig
from src.algorithms.jepa.jepa_config import JepaConfig
from src.algorithms.jepa_decoder.jepa_decoder_config import JepaDecoderConfig

_ALGORITHM_REGISTRY: dict[str, type[AlgorithmConfig]] = {
    "base": AlgorithmConfig,
    "jepa": JepaConfig,
    "jepa_decoder": JepaDecoderConfig
}

@dataclass
class RunInfoSupervised(RunInfo):
    dataset : str

    def tags(self) -> list[str]:
        return [self.algorithm_id[-64:], self.task_id[-64:], self.dataset[-64:]]




@dataclass
class RunConfig:
    mdp: MdpConfig = field(default_factory=MdpConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)


def load_config(path: str, algorithm_id: str) -> RunConfig:
    sections = {
        "mdp": MdpConfig,
        "algorithm": _ALGORITHM_REGISTRY[algorithm_id]
    }
    return RunConfig(**ConfigLoader.load_single(path, sections))


def load_grid_configs(path: str, algorithm_id: str) -> list[RunConfig]:
    sections = {
        "mdp": MdpConfig,
        "algorithm": _ALGORITHM_REGISTRY[algorithm_id]
    }
    return [RunConfig(**d) for d in ConfigLoader.load_grid(path, sections)]

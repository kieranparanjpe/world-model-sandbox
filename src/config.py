from __future__ import annotations

from dataclasses import dataclass, field

from rl_commons.config import RunInfo, ConfigLoader
from rl_commons.mdp import MdpConfig

# Re-export so callers that import RunInfo/ConfigLoader from src.config continue to work.
__all__ = ["RunConfig", "load_config", "load_grid_configs"]


@dataclass
class RunConfig:
    mdp: MdpConfig = field(default_factory=MdpConfig)


def load_config(path: str, algorithm_id: str, policy_id: str) -> RunConfig:
    sections = {
        "mdp": MdpConfig,
    }
    return RunConfig(**ConfigLoader.load_single(path, sections))


def load_grid_configs(path: str, algorithm_id: str, policy_id: str) -> list[RunConfig]:
    sections = {
        "mdp": MdpConfig,
    }
    return [RunConfig(**d) for d in ConfigLoader.load_grid(path, sections)]

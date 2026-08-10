from typing import Callable

from src.algorithms.jepa.jepa import jepa_factory

ALGORITHM_MAPPING : dict[str, Callable] = {
    'jepa': jepa_factory
}

def create_algorithm(algorithm_id : str, **kwargs):
    return ALGORITHM_MAPPING[algorithm_id](**kwargs)
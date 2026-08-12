import inspect
from typing import Callable

from src.algorithms.jepa.jepa import jepa_factory
from src.algorithms.jepa_decoder.jepa_decoder import jepa_decoder_factory

ALGORITHM_MAPPING : dict[str, Callable] = {
    'jepa': jepa_factory,
    'jepa_decoder': jepa_decoder_factory
}

def create_algorithm(algorithm_id : str, **kwargs):
    factory = ALGORITHM_MAPPING[algorithm_id]
    
    # 1. Ask Python what arguments this specific factory accepts
    sig = inspect.signature(factory)
    valid_keys = sig.parameters.keys()
    
    # 2. Filter your kwargs to only include things the factory asked for
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    # 3. Pass the cleaned dictionary
    return factory(**filtered_kwargs)
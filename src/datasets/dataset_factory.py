from typing import Callable

from src.algorithms.jepa.jepa import jepa_factory
from src.algorithms.jepa_decoder.jepa_decoder import jepa_decoder_factory
from src.datasets.dataset_encoder import DatasetEncoder
from src.datasets.dataset_sa import DatasetSA

DATASET_MAPPING : dict[str, Callable] = {
    'sa': DatasetSA,
    'encoder': DatasetEncoder
}

def create_dataset(dataset_id : str, **kwargs):
    return DATASET_MAPPING[dataset_id](**kwargs)
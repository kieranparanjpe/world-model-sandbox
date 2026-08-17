import torch
from ml_commons.networks import NetworkConfig


class DecoderConfig(NetworkConfig):
    encoding_space_size : int = 128


torch.serialization.add_safe_globals([DecoderConfig])
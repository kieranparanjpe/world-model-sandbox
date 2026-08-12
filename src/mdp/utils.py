import torch
from rl_commons.mdp import Mdp


def get_action_sampler(_mdp : Mdp):
    if _mdp.discrete:
        return torch.distributions.Categorical(logits=torch.ones(_mdp.action_dimension))
    else:
        return torch.distributions.Uniform(low=_mdp.action_range[:, 0], high=_mdp.action_range[:, 1])
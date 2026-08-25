from rl_commons.mdp import Mdp
from rl_commons.policies import Policy
from my_rl_impl.algorithms.policies import CategoricalPolicyConfig

from src.mdp.uniform_policy import UniformPolicyConfig


def get_random_policy(mdp : Mdp) -> Policy:
    if mdp.discrete:
        policy_id = "categorical"
        policy_config = CategoricalPolicyConfig()
        policy = Policy.build_policy(policy_id, mdp.obs_dimension, mdp.action_dimension, policy_config)
    else:
        policy_id = "uniform"
        policy_config = UniformPolicyConfig(action_range=(float(mdp.action_range[0][0]), float(mdp.action_range[0][1])))
        policy = Policy.build_policy(policy_id, mdp.obs_dimension, mdp.action_dimension, policy_config)
    return policy.to(mdp.device)




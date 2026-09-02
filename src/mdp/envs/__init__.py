import gymnasium

from .static_lunar_lander import StaticLunarLander  # noqa: F401

gymnasium.register(
    id="StaticLunarLander-v0",
    entry_point="src.mdp.envs.static_lunar_lander:StaticLunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

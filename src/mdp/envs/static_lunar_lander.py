import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander


class StaticLunarLander(LunarLander):
    """LunarLander without the random initial impulse applied on reset()."""

    def reset(self, **kwargs):
        # INITIAL_RANDOM is a module-level constant read by reset() as a bare global,
        # not an instance/class attribute, so it must be patched at the module level.
        original_initial_random = gym.envs.box2d.lunar_lander.INITIAL_RANDOM
        gym.envs.box2d.lunar_lander.INITIAL_RANDOM = 0.0
        try:
            return super().reset(**kwargs)
        finally:
            gym.envs.box2d.lunar_lander.INITIAL_RANDOM = original_initial_random

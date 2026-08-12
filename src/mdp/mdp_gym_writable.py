from typing import Callable
import sys

import torch
from rl_commons.mdp import MdpGym

from src.mdp.mdp_writable import MdpWritable


class MdpGymWritable(MdpWritable, MdpGym):

    def __init__(self, *args, **kwargs):
        MdpGym.__init__(self, *args, **kwargs)
        self.mdp_set_state_dispatch: dict[str, Callable[[torch.Tensor], None]] = {
            "LunarLander": self._set_lunar_lander_state
        }
        self.mdp_type_set_state_dispatch: dict[str, Callable[[torch.Tensor], None]] = {
            "mujoco": self._set_mujoco_state
        }

    def _set_mujoco_state(self, state: torch.Tensor):
        env = self._env.unwrapped
        np_state = state.cpu().numpy()

        # MuJoCo environments store the number of position variables in env.model.nq
        nq = env.model.nq

        # The state is always [qpos, qvel] concatenated together
        qpos = np_state[:nq]
        qvel = np_state[nq:]

        env.set_state(qpos, qvel)

    def _set_lunar_lander_state(self, state: torch.Tensor):
        env = self._env.unwrapped
        np_state = state.cpu().numpy()

        # Dynamically fetch the module where this environment was defined
        ll_module = sys.modules[env.__class__.__module__]

        # Extract the exact constants from the module
        VIEWPORT_W = ll_module.VIEWPORT_W
        VIEWPORT_H = ll_module.VIEWPORT_H
        SCALE = ll_module.SCALE
        FPS = ll_module.FPS
        LEG_DOWN = ll_module.LEG_DOWN

        # Reverse the scaling applied in LunarLander's step() function
        # Note: env.helipad_y is dynamically calculated and stored on the env object during env.reset()
        x = np_state[0] * (VIEWPORT_W / SCALE / 2) + (VIEWPORT_W / SCALE / 2)
        y = np_state[1] * (VIEWPORT_H / SCALE / 2) + (env.helipad_y + LEG_DOWN / SCALE)
        vx = np_state[2] * FPS / (VIEWPORT_W / SCALE / 2)
        vy = np_state[3] * FPS / (VIEWPORT_H / SCALE / 2)
        angle = np_state[4]
        angular_velocity = np_state[5] * FPS / 20.0

        # Teleport the main Box2D lander body
        env.lander.position = (x, y)
        env.lander.linearVelocity = (vx, vy)
        env.lander.angle = angle
        env.lander.angularVelocity = angular_velocity

    def set_state(self, state : torch.Tensor):
        for key, function in self.mdp_set_state_dispatch.items():
            if key in self._env.unwrapped.spec.id:
                return function(state)

        for key, function in self.mdp_type_set_state_dispatch.items():
            if key in self._env.unwrapped.__class__.__module__:
                return function(state)

        return None


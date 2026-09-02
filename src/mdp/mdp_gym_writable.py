from typing import Callable, Optional

import numpy as np
import torch
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module
from rl_commons.mdp import MdpGym, MdpTerminationState

from src.mdp import envs  # noqa: F401 — registers custom environments with gymnasium
from src.mdp.mdp_writable import MdpWritable


class MdpGymWritable(MdpWritable, MdpGym):

    # obs = [qpos[n_excluded:], qvel, ...ignored extra features]: these env families exclude the
    # first n_excluded qpos dims (root x[,y] position) from the observation entirely, since it's
    # translation-irrelevant to the policy. See each env's _get_obs() in gymnasium.envs.mujoco.
    _MUJOCO_EXCLUDED_QPOS_DIMS: dict[str, int] = {
        "HopperEnv": 1,
        "Walker2dEnv": 1,
        "HalfCheetahEnv": 1,
        "AntEnv": 2,
        "HumanoidEnv": 2,
    }

    def __init__(self, *args, **kwargs):
        MdpGym.__init__(self, *args, **kwargs)
        self._reconstructed_root_pos: np.ndarray | None = None
        self.mdp_set_state_dispatch: dict[str, Callable[[torch.Tensor], MdpTerminationState]] = {
            "LunarLander": self._set_lunar_lander_state
        }
        self.mdp_type_set_state_dispatch: dict[str, Callable[[torch.Tensor], MdpTerminationState]] = {
            "mujoco": self._set_mujoco_state
        }

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        self._reconstructed_root_pos = None
        return MdpGym.reset(self, seed=seed)

    def render(self):
        return self._env.render()

    def _set_mujoco_state(self, state: torch.Tensor) -> MdpTerminationState:
        env = self._env.unwrapped
        cls_name = env.__class__.__name__
        if cls_name not in self._MUJOCO_EXCLUDED_QPOS_DIMS:
            raise NotImplementedError(
                f"MdpGymWritable._set_mujoco_state has no dispatch for MuJoCo env class {cls_name!r}"
            )
        n_excluded = self._MUJOCO_EXCLUDED_QPOS_DIMS[cls_name]

        np_state = state.cpu().numpy()
        nq, nv = env.model.nq, env.model.nv

        qpos_partial = np_state[: nq - n_excluded]
        qvel = np_state[nq - n_excluded : nq - n_excluded + nv]

        # The excluded root position isn't in the observation at all, so it can't be read back --
        # reconstruct it by integrating the (included) root velocity across calls, seeded from the
        # real position at the last reset().
        if self._reconstructed_root_pos is None:
            self._reconstructed_root_pos = env.data.qpos[:n_excluded].copy()
        else:
            self._reconstructed_root_pos = self._reconstructed_root_pos + qvel[:n_excluded] * env.dt

        qpos = np.concatenate([self._reconstructed_root_pos, qpos_partial])

        # set_state() calls mujoco.mj_forward() internally, so is_healthy/site_xpos/etc.
        # below are already up to date for the new state -- no simulation step needed.
        env.set_state(qpos, qvel)

        return MdpTerminationState.TERMINATED if self._is_mujoco_terminated(env) else MdpTerminationState.IN_PROGRESS

    def _is_mujoco_terminated(self, env) -> bool:
        # Ant/Hopper/Walker2d/Humanoid (v4 & v5) expose this as a pure function of qpos/qvel
        if hasattr(env, "is_healthy"):
            return (not env.is_healthy) and getattr(env, "_terminate_when_unhealthy", True)

        cls_name = env.__class__.__name__
        if cls_name == "InvertedPendulumEnv":
            obs = env._get_obs()
            return bool(not np.isfinite(obs).all() or abs(obs[1]) > 0.2)
        if cls_name == "InvertedDoublePendulumEnv":
            return bool(env.data.site_xpos[0][2] <= 1)

        # HalfCheetah, HumanoidStandup, Pusher, Reacher, Swimmer have no early termination
        return False

    def _set_lunar_lander_state(self, state: torch.Tensor) -> MdpTerminationState:
        env = self._env.unwrapped
        np_state = state.cpu().numpy()

        # Extract the exact constants from the module, even if env is a LunarLander subclass
        # (e.g. StaticLunarLander) that doesn't redefine them.
        VIEWPORT_W = lunar_lander_module.VIEWPORT_W
        VIEWPORT_H = lunar_lander_module.VIEWPORT_H
        SCALE = lunar_lander_module.SCALE
        FPS = lunar_lander_module.FPS
        LEG_DOWN = lunar_lander_module.LEG_DOWN
        LEG_AWAY = lunar_lander_module.LEG_AWAY

        # Reverse the scaling applied in LunarLander's step() function
        # Note: env.helipad_y is dynamically calculated and stored on the env object during env.reset()
        # pybox2d's setters reject numpy scalars (e.g. float32 from the obs array), so cast to float.
        x = float(np_state[0] * (VIEWPORT_W / SCALE / 2) + (VIEWPORT_W / SCALE / 2))
        y = float(np_state[1] * (VIEWPORT_H / SCALE / 2) + (env.helipad_y + LEG_DOWN / SCALE))
        vx = float(np_state[2] * FPS / (VIEWPORT_W / SCALE / 2))
        vy = float(np_state[3] * FPS / (VIEWPORT_H / SCALE / 2))
        angle = float(np_state[4])
        angular_velocity = float(np_state[5] * FPS / 20.0)

        # Teleport the main Box2D lander body
        env.lander.position = (x, y)
        env.lander.linearVelocity = (vx, vy)
        env.lander.angle = angle
        env.lander.angularVelocity = angular_velocity

        # Legs are separate Box2D bodies joined to the lander by a revolute joint: the joint
        # only constrains the pivot point (leg's local anchor coincides with the lander's
        # origin) -- rotation is a free DOF (motor + limits), not derivable from the lander's
        # pose. Re-solve that same anchor equation for the new lander position, at each leg's
        # current angle, instead of leaving legs stranded wherever they last physically settled.
        for leg, i in zip(env.legs, (-1, 1)):
            anchor_b_x, anchor_b_y = i * LEG_AWAY / SCALE, LEG_DOWN / SCALE
            cos_leg, sin_leg = np.cos(leg.angle), np.sin(leg.angle)
            anchor_world_x = cos_leg * anchor_b_x - sin_leg * anchor_b_y
            anchor_world_y = sin_leg * anchor_b_x + cos_leg * anchor_b_y
            leg.position = (x - anchor_world_x, y - anchor_world_y)

        # Box2D only resolves contacts (and thus game_over/leg ground_contact) inside world.Step().
        # A zero-length step forces contact resolution at the new position without advancing
        # simulated time. NOTE: this can't recover "awake" (landed-and-settled) -- Box2D's sleep
        # state requires ~0.5s of sustained low velocity accumulated across real steps, which a
        # teleported snapshot fundamentally can't provide. Only crash/out-of-bounds are detected.
        env.world.Step(0, 6 * 30, 2 * 30)

        out_of_bounds = abs(np_state[0]) >= 1.0
        return MdpTerminationState.TERMINATED if (env.game_over or out_of_bounds) else MdpTerminationState.IN_PROGRESS

    def set_state(self, state : torch.Tensor) -> MdpTerminationState:
        for key, function in self.mdp_set_state_dispatch.items():
            if key in self._env.unwrapped.spec.id:
                return function(state)

        for key, function in self.mdp_type_set_state_dispatch.items():
            if key in self._env.unwrapped.__class__.__module__:
                return function(state)

        raise NotImplementedError(
            f"MdpGymWritable.set_state() has no dispatch for env {self._env.unwrapped.spec.id!r}"
        )


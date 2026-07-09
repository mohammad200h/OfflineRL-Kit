import numpy as np

from typing import Tuple, Dict


class MujocoOracleDynamics(object):
    def __init__(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

    def _set_state_from_obs(self, obs: np.ndarray) -> None:
        self.env.reset()
        if len(obs) == (self.env.model.nq + self.env.model.nv - 1):
            xpos = np.zeros(1)
            obs = np.concatenate([xpos, obs])
        qpos = obs[: self.env.model.nq]
        qvel = obs[self.env.model.nq :]
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0
        self.env.set_state(qpos, qvel)

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        if (len(obs.shape) > 1) or (len(action.shape) > 1):
            raise ValueError
        self.env.reset()
        self._set_state_from_obs(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return next_obs, reward, terminated or truncated, info
"""
D4RL task helpers built on Gymnasium and the native MuJoCo Python bindings.

Offline datasets are still loaded from the original D4RL HDF5 files, while
evaluation uses modern Gymnasium MuJoCo environments (v5) or Gymnasium-Robotics
for AntMaze tasks. Classic-control demos (MountainCar) use a local HDF5
installed into ``$D4RL_DATASET_DIR``.
"""
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from tqdm import tqdm

try:
    import gymnasium_robotics  # noqa: F401

    _HAS_ROBOTICS = True
except ImportError:
    _HAS_ROBOTICS = False

DATASET_PATH = os.environ.get(
    "D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets")
)

MUJOCO_AGENT_TO_ENV = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "ant": "Ant-v5",
}

ANTMAZE_TASK_TO_ENV = {
    "antmaze-umaze-v2": "AntMaze_UMaze-v4",
    "antmaze-umaze-diverse-v2": "AntMaze_UMaze-v4",
    "antmaze-medium-play-v2": "AntMaze_Medium-v4",
    "antmaze-medium-diverse-v2": "AntMaze_Medium_Diverse_G-v4",
    "antmaze-large-play-v2": "AntMaze_Large-v4",
    "antmaze-large-diverse-v2": "AntMaze_Large_Diverse_G-v4",
}

# Local demo datasets (basename under $D4RL_DATASET_DIR). Install with
# data_colllection/install_mountain_car_dataset.py.
CLASSIC_CONTROL_TASK_TO_ENV = {
    "mountaincar-human-v0": "MountainCar-v0",
}

CLASSIC_CONTROL_MAX_EPISODE_STEPS = {
    "mountaincar-human-v0": 200,
}

REF_MIN_SCORE: Dict[str, float] = {
    "halfcheetah-random-v0": -280.178953,
    "hopper-random-v0": -20.272305,
    "walker2d-random-v0": 1.629008,
    "ant-random-v0": -325.6,
    "mountaincar-human-v0": -200.0,
    "antmaze-umaze-v0": 0.0,
    "antmaze-umaze-diverse-v0": 0.0,
    "antmaze-medium-play-v0": 0.0,
    "antmaze-medium-diverse-v0": 0.0,
    "antmaze-large-play-v0": 0.0,
    "antmaze-large-diverse-v0": 0.0,
    "antmaze-umaze-v2": 0.0,
    "antmaze-umaze-diverse-v2": 0.0,
    "antmaze-medium-play-v2": 0.0,
    "antmaze-medium-diverse-v2": 0.0,
    "antmaze-large-play-v2": 0.0,
    "antmaze-large-diverse-v2": 0.0,
}

REF_MAX_SCORE: Dict[str, float] = {
    "halfcheetah-random-v0": 12135.0,
    "hopper-random-v0": 3234.3,
    "walker2d-random-v0": 4592.3,
    "ant-random-v0": 3879.7,
    "mountaincar-human-v0": -110.0,
    "antmaze-umaze-v0": 1.0,
    "antmaze-umaze-diverse-v0": 1.0,
    "antmaze-medium-play-v0": 1.0,
    "antmaze-medium-diverse-v0": 1.0,
    "antmaze-large-play-v0": 1.0,
    "antmaze-large-diverse-v0": 1.0,
    "antmaze-umaze-v2": 1.0,
    "antmaze-umaze-diverse-v2": 1.0,
    "antmaze-medium-play-v2": 1.0,
    "antmaze-medium-diverse-v2": 1.0,
    "antmaze-large-play-v2": 1.0,
    "antmaze-large-diverse-v2": 1.0,
}

DATASET_URLS: Dict[str, str] = {
    # Basename must match the file placed by install_mountain_car_dataset.py
    "mountaincar-human-v0": "local://mountain_car_human.hdf5",
    "antmaze-umaze-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5",
    "antmaze-umaze-diverse-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    "antmaze-medium-play-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    "antmaze-medium-diverse-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    "antmaze-large-play-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    "antmaze-large-diverse-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
}

for _agent in MUJOCO_AGENT_TO_ENV:
    for _dataset in [
        "random",
        "medium",
        "expert",
        "medium-replay",
        "full-replay",
        "medium-expert",
    ]:
        for _version in ("v1", "v2"):
            _dset_name = f"{_agent}_{_dataset.replace('-', '_')}-{_version}"
            _env_name = _dset_name.replace("_", "-")
            DATASET_URLS[_env_name] = (
                "http://rail.eecs.berkeley.edu/datasets/offline_rl/"
                f"gym_mujoco_{_version}/{_dset_name}.hdf5"
            )
            REF_MIN_SCORE[_env_name] = REF_MIN_SCORE[f"{_agent}-random-v0"]
            REF_MAX_SCORE[_env_name] = REF_MAX_SCORE[f"{_agent}-random-v0"]


def _register_robotics_envs() -> None:
    if _HAS_ROBOTICS:
        gym.register_envs(gymnasium_robotics)


class DiscreteAsBoxWrapper(gym.ActionWrapper):
    """Expose Discrete actions as Box(1,) floats for OfflineRL-Kit policies."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, Discrete):
            raise TypeError(
                f"DiscreteAsBoxWrapper expects Discrete action space, got {env.action_space}"
            )
        self._n = int(env.action_space.n)
        self.action_space = Box(
            low=np.zeros((1,), dtype=np.float32),
            high=np.full((1,), self._n - 1, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, act: np.ndarray) -> int:
        value = float(np.asarray(act, dtype=np.float32).reshape(-1)[0])
        return int(np.clip(np.rint(value), 0, self._n - 1))


def _get_h5_keys(h5file: h5py.File) -> list:
    keys = []

    def visitor(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def _dataset_filepath(dataset_url: str) -> str:
    _, dataset_name = os.path.split(dataset_url)
    return os.path.join(DATASET_PATH, dataset_name)


def download_dataset(dataset_url: str) -> str:
    os.makedirs(DATASET_PATH, exist_ok=True)
    dataset_filepath = _dataset_filepath(dataset_url)
    if not os.path.exists(dataset_filepath):
        if dataset_url.startswith("local://"):
            raise IOError(
                f"Local dataset missing: {dataset_filepath}. "
                "Install it with: "
                "python3 data_colllection/install_mountain_car_dataset.py"
            )
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError(f"Failed to download dataset from {dataset_url}")
    return dataset_filepath


def _resolve_gymnasium_env_id(task: str) -> str:
    if task in CLASSIC_CONTROL_TASK_TO_ENV:
        return CLASSIC_CONTROL_TASK_TO_ENV[task]

    if task in ANTMAZE_TASK_TO_ENV:
        if not _HAS_ROBOTICS:
            raise ImportError(
                f"Task '{task}' requires gymnasium-robotics. "
                "Install it with: pip install gymnasium-robotics"
            )
        return ANTMAZE_TASK_TO_ENV[task]

    for agent, env_id in MUJOCO_AGENT_TO_ENV.items():
        if task.startswith(agent + "-"):
            return env_id

    raise ValueError(
        f"Unsupported D4RL task '{task}'. Supported families: "
        f"{list(MUJOCO_AGENT_TO_ENV)}, {list(ANTMAZE_TASK_TO_ENV)}, "
        f"and {list(CLASSIC_CONTROL_TASK_TO_ENV)}."
    )


class D4RLEnv:
    """Gymnasium evaluation env with D4RL dataset and score normalization helpers."""

    def __init__(
        self,
        task: str,
        env: Env,
        dataset_url: str,
        ref_min_score: float,
        ref_max_score: float,
        max_episode_steps: int = 1000,
    ) -> None:
        self.task = task
        self.env = env
        self._dataset_url = dataset_url
        self.ref_min_score = ref_min_score
        self.ref_max_score = ref_max_score
        self._max_episode_steps = max_episode_steps

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def seed(self, seed: int) -> None:
        set_env_seed(self, seed)

    def get_normalized_score(self, score: float) -> float:
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_dataset(self, h5path: Optional[str] = None) -> Dict[str, np.ndarray]:
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset(self._dataset_url)

        data_dict: Dict[str, np.ndarray] = {}
        with h5py.File(h5path, "r") as dataset_file:
            for key in tqdm(_get_h5_keys(dataset_file), desc="load datafile"):
                try:
                    data_dict[key] = dataset_file[key][:]
                except ValueError:
                    data_dict[key] = dataset_file[key][()]

        for key in ["observations", "actions", "rewards", "terminals"]:
            if key not in data_dict:
                raise KeyError(f"Dataset is missing key {key}")

        n_samples = data_dict["observations"].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict["observations"].shape[1:] == self.observation_space.shape
        assert data_dict["actions"].shape[1:] == self.action_space.shape

        if data_dict["rewards"].shape == (n_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]
        if data_dict["terminals"].shape == (n_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]

        return data_dict


def make_env(task: str, max_episode_steps: int = 1000) -> D4RLEnv:
    if task not in DATASET_URLS:
        raise ValueError(f"Unknown D4RL task '{task}'.")

    _register_robotics_envs()
    env_id = _resolve_gymnasium_env_id(task)
    steps = CLASSIC_CONTROL_MAX_EPISODE_STEPS.get(task, max_episode_steps)
    env = gym.make(env_id, max_episode_steps=steps)
    if task in CLASSIC_CONTROL_TASK_TO_ENV:
        # Demos store discrete actions as shape (N, 1) floats; OfflineRL-Kit
        # also expects action_space.high for continuous policy heads.
        env = DiscreteAsBoxWrapper(env)

    return D4RLEnv(
        task=task,
        env=env,
        dataset_url=DATASET_URLS[task],
        ref_min_score=REF_MIN_SCORE[task],
        ref_max_score=REF_MAX_SCORE[task],
        max_episode_steps=steps,
    )


def set_env_seed(env: Union[D4RLEnv, Env], seed: int) -> None:
    target = env.env if isinstance(env, D4RLEnv) else env
    target.reset(seed=seed)


def qlearning_dataset(
    env: D4RLEnv,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Format a D4RL dataset for Q-learning style offline RL algorithms."""
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    has_next_obs = "next_observations" in dataset
    n = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], []
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(n - 1):
        obs = dataset["observations"][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset["next_observations"][i].astype(np.float32)
        else:
            new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        if (not terminate_on_end) and final_timestep:
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }

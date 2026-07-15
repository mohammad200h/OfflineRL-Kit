"""Collect MountainCarContinuous-v0 demos with keyboard and save a D4RL-style HDF5.

Controls
--------
Left arrow  : ramp force toward -1.0 (gradual)
Right arrow : ramp force toward +1.0 (gradual)
(no key)    : coast force toward 0.0
R           : end the current episode early and start a new one
S           : save dataset to disk (keep playing)
Q / Esc     : save and quit

Inspect saved data
------------------
python play.py --view
python play.py --view --output path/to/mountain_car_human.hdf5
python play.py --delete incomplete
python play.py --delete timeout incomplete

The saved file matches OfflineRL-Kit's expected offline dataset keys so it can
be passed into ``qlearning_dataset`` / ``ReplayBuffer.load_dataset`` used by
``run_example/run_combo.py``:

    import h5py
    from offlinerlkit.utils.load_dataset import qlearning_dataset

    with h5py.File("demostrations/mountain_car_human.hdf5", "r") as f:
        raw = {k: f[k][:] for k in f.keys()}
    dataset = qlearning_dataset(env, dataset=raw)
    real_buffer.load_dataset(dataset)

Env reference:
https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import h5py
import numpy as np

from controll import SpeedController, speed_bar


ENV_ID = "MountainCarContinuous-v0"


DATASET_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "next_observations",
)


def _empty_buffers() -> Dict[str, List[np.ndarray]]:
    return {k: [] for k in DATASET_KEYS}


def load_d4rl_hdf5(path: str) -> Dict[str, List[np.ndarray]]:
    """Load an existing dataset into list buffers (for appending)."""
    buffers = _empty_buffers()
    if not os.path.isfile(path):
        return buffers

    with h5py.File(path, "r") as f:
        missing = [k for k in DATASET_KEYS if k not in f]
        if missing:
            raise KeyError(
                f"Existing dataset {path} is missing keys {missing}; "
                "refuse to append."
            )
        for key in DATASET_KEYS:
            data = f[key][:]
            if key == "actions":
                data = np.asarray(data, dtype=np.float32).reshape(-1)
            for row in data:
                buffers[key].append(row)

    n = len(buffers["rewards"])
    print(f"Loaded {n} existing transitions from {path} (will append).")
    return buffers


def load_d4rl_arrays(path: str) -> Dict[str, np.ndarray]:
    """Load dataset as numpy arrays (read-only view helper)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No dataset found at {path}")
    with h5py.File(path, "r") as f:
        missing = [k for k in DATASET_KEYS if k not in f]
        if missing:
            raise KeyError(f"Dataset {path} is missing keys {missing}")
        data = {k: f[k][:] for k in DATASET_KEYS}
    data["actions"] = np.asarray(data["actions"], dtype=np.float32).reshape(-1)
    data["rewards"] = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    data["terminals"] = np.asarray(data["terminals"], dtype=np.bool_).reshape(-1)
    data["timeouts"] = np.asarray(data["timeouts"], dtype=np.bool_).reshape(-1)
    return data


def split_trajectories(
    data: Dict[str, np.ndarray],
) -> List[Tuple[int, int, int, str, float]]:
    """Split the flat dataset into trajectories.

    Returns a list of (traj_id, start, end, status, episode_return)
    where ``end`` is exclusive and status is ``success``, ``timeout``,
    or ``incomplete``.
    """
    n = len(data["rewards"])
    trajs: List[Tuple[int, int, int, str, float]] = []
    start = 0
    traj_id = 0
    for i in range(n):
        done = bool(data["terminals"][i]) or bool(data["timeouts"][i])
        if not done:
            continue
        end = i + 1
        ep_return = float(np.sum(data["rewards"][start:end]))
        if bool(data["terminals"][i]):
            status = "success"
        else:
            status = "timeout"
        traj_id += 1
        trajs.append((traj_id, start, end, status, ep_return))
        start = end

    if start < n:
        ep_return = float(np.sum(data["rewards"][start:n]))
        traj_id += 1
        trajs.append((traj_id, start, n, "incomplete", ep_return))
    return trajs


def view_dataset(path: str) -> None:
    """Print trajectory summary: length, return, success/timeout/incomplete."""
    data = load_d4rl_arrays(path)
    trajs = split_trajectories(data)
    n_trans = len(data["rewards"])
    n_success = sum(1 for t in trajs if t[3] == "success")
    n_timeout = sum(1 for t in trajs if t[3] == "timeout")
    n_incomplete = sum(1 for t in trajs if t[3] == "incomplete")

    print(f"Dataset: {path}")
    print(f"Transitions: {n_trans}")
    print(
        f"Trajectories: {len(trajs)}  "
        f"(success={n_success}, timeout={n_timeout}, incomplete={n_incomplete})"
    )
    print("-" * 60)
    print(f"{'#':>4}  {'status':<12}  {'len':>5}  {'return':>8}")
    print("-" * 60)
    for traj_id, start, end, status, ep_return in trajs:
        print(f"{traj_id:>4}  {status:<12}  {end - start:>5}  {ep_return:>8.0f}")
    print("-" * 60)


def save_d4rl_arrays(path: str, data: Dict[str, np.ndarray]) -> int:
    """Overwrite HDF5 from numpy arrays (actions stored as (N, 1))."""
    n = len(data["rewards"])
    if n == 0:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted empty dataset file → {path}")
        else:
            print("Nothing left to save.")
        return 0

    observations = np.asarray(data["observations"], dtype=np.float32)
    next_observations = np.asarray(data["next_observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32).reshape(-1, 1)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    terminals = np.asarray(data["terminals"], dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(data["timeouts"], dtype=np.bool_).reshape(-1)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=observations, compression="gzip")
        f.create_dataset("next_observations", data=next_observations, compression="gzip")
        f.create_dataset("actions", data=actions, compression="gzip")
        f.create_dataset("rewards", data=rewards, compression="gzip")
        f.create_dataset("terminals", data=terminals, compression="gzip")
        f.create_dataset("timeouts", data=timeouts, compression="gzip")
    return n


def delete_trajectories(path: str, statuses: List[str]) -> None:
    """Remove all trajectories whose status is in ``statuses`` and rewrite HDF5."""
    statuses_set = set(statuses)
    data = load_d4rl_arrays(path)
    trajs = split_trajectories(data)

    keep_mask = np.ones(len(data["rewards"]), dtype=bool)
    removed = []
    for traj_id, start, end, status, ep_return in trajs:
        if status in statuses_set:
            keep_mask[start:end] = False
            removed.append((traj_id, end - start, status, ep_return))

    if not removed:
        print(f"No trajectories with status in {sorted(statuses_set)} to delete.")
        return

    kept = {k: np.asarray(v)[keep_mask] for k, v in data.items()}
    n_before = len(data["rewards"])
    n_after = save_d4rl_arrays(path, kept)

    print(f"Deleted {len(removed)} trajectories ({n_before - n_after} transitions):")
    for traj_id, length, status, ep_return in removed:
        print(f"  #{traj_id}  {status:<12}  len={length}  return={ep_return:.0f}")
    print(f"Remaining: {n_after} transitions → {path}")


def save_d4rl_hdf5(
    path: str,
    buffers: Dict[str, List[np.ndarray]],
    *,
    prev_count: int = 0,
) -> int:
    """Write a D4RL-compatible HDF5 dataset for OfflineRL-Kit.

    When ``path`` already exists, callers should have loaded it into
    ``buffers`` first so this rewrite effectively appends new transitions.
    """
    if not buffers["observations"]:
        print("No transitions collected yet; nothing to save.")
        return 0

    observations = np.asarray(buffers["observations"], dtype=np.float32)
    next_observations = np.asarray(buffers["next_observations"], dtype=np.float32)
    # Shape (N, 1) so ReplayBuffer / COMBO treat action_dim=1 cleanly.
    actions = np.asarray(buffers["actions"], dtype=np.float32).reshape(-1, 1)
    rewards = np.asarray(buffers["rewards"], dtype=np.float32)
    terminals = np.asarray(buffers["terminals"], dtype=np.bool_)
    timeouts = np.asarray(buffers["timeouts"], dtype=np.bool_)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=observations, compression="gzip")
        f.create_dataset("next_observations", data=next_observations, compression="gzip")
        f.create_dataset("actions", data=actions, compression="gzip")
        f.create_dataset("rewards", data=rewards, compression="gzip")
        f.create_dataset("terminals", data=terminals, compression="gzip")
        f.create_dataset("timeouts", data=timeouts, compression="gzip")

    n = len(rewards)
    n_new = n - prev_count
    n_success = int(terminals.sum())
    if prev_count > 0:
        print(
            f"Saved {n} transitions "
            f"(+{n_new} new, {n_success} successful terminals) → {path}"
        )
    else:
        print(
            f"Saved {n} transitions ({n_success} successful terminals) → {path}"
        )
    return n


def play(args: argparse.Namespace) -> None:
    env = gym.make(ENV_ID, render_mode="human")
    controller = SpeedController(
        max_speed=args.max_speed,
        acceleration=args.accel,
        deceleration=args.decel,
    )
    controller.start()

    buffers = load_d4rl_hdf5(args.output)
    prev_count = len(buffers["rewards"])
    episode = 0
    episode_return = 0.0
    episode_len = 0
    total_steps = 0
    dt = args.step_delay

    obs, _ = env.reset(seed=args.seed)
    episode += 1
    print(
        f"{ENV_ID} keyboard collection (gradual throttle)\n"
        "  ← / → : ramp force toward -1 / +1\n"
        "  (release to coast toward 0)\n"
        "  R : reset episode   S : save   Q/Esc : save & quit\n"
        f"  accel={args.accel}/s  decel={args.decel}/s  max={args.max_speed}\n"
    )

    try:
        while not controller.quit_requested:
            if controller.reset_requested:
                controller.reset_requested = False
                controller.reset()
                print(
                    f"\n[ep {episode}] aborted  return={episode_return:.1f}  "
                    f"len={episode_len}"
                )
                obs, _ = env.reset()
                episode += 1
                episode_return = 0.0
                episode_len = 0
                continue

            if controller.save_requested:
                controller.save_requested = False
                print()  # keep speed line from being overwritten by save msg
                save_d4rl_hdf5(args.output, buffers, prev_count=prev_count)

            speed = controller.update(dt)
            action = controller.get_action()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            buffers["observations"].append(np.asarray(obs, dtype=np.float32))
            buffers["actions"].append(np.float32(action[0]))
            buffers["rewards"].append(np.float32(reward))
            buffers["terminals"].append(bool(terminated))
            buffers["timeouts"].append(bool(truncated))
            buffers["next_observations"].append(
                np.asarray(next_obs, dtype=np.float32)
            )

            obs = next_obs
            episode_return += float(reward)
            episode_len += 1
            total_steps += 1

            bar = speed_bar(speed, args.max_speed)
            sys.stdout.write(
                f"\r  ep={episode}  step={episode_len}  "
                f"ret={episode_return:+7.1f}  "
                f"force={speed:+6.3f}  {bar}  "
            )
            sys.stdout.flush()

            if terminated or truncated:
                status = "SUCCESS" if terminated else "timeout"
                print(
                    f"\n[ep {episode}] {status}  return={episode_return:.1f}  "
                    f"len={episode_len}  total_steps={total_steps}"
                )
                controller.reset()
                obs, _ = env.reset()
                episode += 1
                episode_return = 0.0
                episode_len = 0

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print()
        save_d4rl_hdf5(args.output, buffers, prev_count=prev_count)
        controller.stop()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect MountainCarContinuous-v0 demos with keyboard for OfflineRL-Kit."
        )
    )
    demo_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "demostrations",
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(demo_dir, "mountain_car_human.hdf5"),
        help="Path to output D4RL-style HDF5 file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.05,
        help="Seconds to sleep between env steps (also throttle dt).",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.0,
        help="Max |force| from SpeedController (default: 1.0).",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=1.5,
        help="Throttle acceleration while holding ←/→, units per second.",
    )
    parser.add_argument(
        "--decel",
        type=float,
        default=2.0,
        help="Coast-down rate toward 0 when no key is held, units per second.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="List trajectories in --output and whether each was successful, then exit.",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        choices=["success", "timeout", "incomplete"],
        metavar="STATUS",
        help=(
            "Delete all trajectories with the given status(es) from --output, "
            "then exit. Example: --delete incomplete timeout"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.delete:
        delete_trajectories(args.output, args.delete)
    elif args.view:
        view_dataset(args.output)
    else:
        play(args)

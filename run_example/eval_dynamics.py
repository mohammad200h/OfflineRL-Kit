"""Evaluate ensemble dynamics by comparing predicted vs true state deltas and rewards.

Samples trajectories from a D4RL-style demo HDF5 (e.g. mountain_car_human.hdf5),
computes true delta = next_obs - obs and true reward from the recorded
transitions, and compares against the dynamics model's mean predictions.

Can be used standalone or as a training callback from ``run_dynamics.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn


DATASET_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "next_observations",
)

DEFAULT_DEMO_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_colllection",
        "demostrations",
        "mountain_car_human.hdf5",
    )
)


def load_demo_arrays(path: str) -> Dict[str, np.ndarray]:
    """Load a D4RL-style demo HDF5 as numpy arrays."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No dataset found at {path}")
    with h5py.File(path, "r") as f:
        missing = [k for k in DATASET_KEYS if k not in f]
        if missing:
            raise KeyError(f"Dataset {path} is missing keys {missing}")
        data = {k: f[k][:] for k in DATASET_KEYS}

    data["observations"] = np.asarray(data["observations"], dtype=np.float32)
    data["next_observations"] = np.asarray(data["next_observations"], dtype=np.float32)
    data["actions"] = np.asarray(data["actions"], dtype=np.float32)
    if data["actions"].ndim == 1:
        data["actions"] = data["actions"].reshape(-1, 1)
    data["rewards"] = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    data["terminals"] = np.asarray(data["terminals"], dtype=np.bool_).reshape(-1)
    data["timeouts"] = np.asarray(data["timeouts"], dtype=np.bool_).reshape(-1)
    return data


def split_trajectories(
    data: Dict[str, np.ndarray],
) -> List[Tuple[int, int]]:
    """Split flat transitions into (start, end) index ranges (end exclusive)."""
    n = len(data["rewards"])
    trajs: List[Tuple[int, int]] = []
    start = 0
    for i in range(n):
        done = bool(data["terminals"][i]) or bool(data["timeouts"][i])
        if done:
            trajs.append((start, i + 1))
            start = i + 1
    if start < n:
        trajs.append((start, n))
    return trajs


def sample_trajectory_indices(
    trajs: List[Tuple[int, int]],
    num_trajs: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Sample up to ``num_trajs`` trajectory spans (without replacement)."""
    if not trajs:
        raise ValueError("No trajectories found in dataset")
    n = min(num_trajs, len(trajs))
    idxs = rng.choice(len(trajs), size=n, replace=False)
    return [trajs[i] for i in idxs]


@torch.no_grad()
def predict_mean_outputs(
    dynamics: EnsembleDynamics,
    obs: np.ndarray,
    action: np.ndarray,
    *,
    use_elites: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Predict mean state delta and reward from the ensemble.

    Returns ``(delta, reward, reward_logits)``.
    ``reward_logits`` is set when the model uses symlog twohot reward heads.
    """
    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(-1, 1)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        action = action.reshape(1, -1)

    obs_act = np.concatenate([obs, action], axis=-1)
    expected = int(np.asarray(dynamics.scaler.mu).reshape(-1).shape[0])
    if obs_act.shape[-1] != expected:
        raise ValueError(
            f"Eval obs+act width {obs_act.shape[-1]} does not match dynamics "
            f"scaler width {expected}. Use the offline dataset for this task "
            f"(not a different env's demos) via --eval-demo-path."
        )
    obs_act = dynamics.scaler.transform(obs_act)

    if dynamics._twohot:
        delta_mean, _, reward_logits = dynamics.model(obs_act)
        if use_elites:
            elite_idxs = dynamics.model.elites.data.cpu().numpy()
            delta_mean = delta_mean[elite_idxs]
            reward_logits = reward_logits[elite_idxs]
        pred_delta = delta_mean.mean(dim=0).cpu().numpy().astype(np.float32)
        logits = reward_logits.mean(dim=0).cpu().numpy().astype(np.float32)
        reward = dynamics.reward_config.decode_logits_to_normalized(logits).reshape(-1)
        return pred_delta, reward, logits

    mean, _ = dynamics.model(obs_act)
    if use_elites:
        elite_idxs = dynamics.model.elites.data.cpu().numpy()
        mean = mean[elite_idxs]
    pred = mean.mean(dim=0).cpu().numpy().astype(np.float32)
    delta = pred[..., :-1]
    reward = pred[..., -1]
    return delta, reward, None


@torch.no_grad()
def predict_delta(
    dynamics: EnsembleDynamics,
    obs: np.ndarray,
    action: np.ndarray,
    *,
    use_elites: bool = True,
) -> np.ndarray:
    """Predict mean state delta (batch, obs_dim) from the ensemble."""
    delta, _, _ = predict_mean_outputs(
        dynamics, obs, action, use_elites=use_elites
    )
    return delta


def evaluate_delta_on_transitions(
    dynamics: EnsembleDynamics,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare true vs predicted deltas (and rewards, if given) on transitions."""
    true_delta = (next_observations - observations).astype(np.float32)
    pred_delta, pred_reward, pred_reward_logits = predict_mean_outputs(
        dynamics, observations, actions
    )

    err = pred_delta - true_delta
    mse_per_dim = np.mean(err ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(err), axis=0)
    mse = float(np.mean(mse_per_dim))
    mae = float(np.mean(mae_per_dim))

    out: Dict[str, Any] = {
        "true_delta": true_delta,
        "pred_delta": pred_delta,
        "error": err,
        "mse": mse,
        "mae": mae,
        "mse_per_dim": mse_per_dim,
        "mae_per_dim": mae_per_dim,
        "pred_reward": pred_reward,
        "n_transitions": int(len(observations)),
    }

    if rewards is not None:
        true_reward = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if true_reward.shape[0] != pred_reward.shape[0]:
            raise ValueError(
                f"rewards length {true_reward.shape[0]} != "
                f"predictions {pred_reward.shape[0]}"
            )
        reward_err = pred_reward - true_reward
        out["true_reward"] = true_reward
        out["reward_error"] = reward_err
        out["reward_mse"] = float(np.mean(reward_err ** 2))
        out["reward_mae"] = float(np.mean(np.abs(reward_err)))
        out["reward_true_mean"] = float(np.mean(true_reward))
        out["reward_pred_mean"] = float(np.mean(pred_reward))

        if dynamics._twohot and pred_reward_logits is not None:
            encoder = dynamics.reward_config.encoder
            preprocessor = dynamics.reward_config.preprocessor
            true_reward_norm = preprocessor.transform(true_reward)
            out["true_reward_norm"] = true_reward_norm
            out["reward_mse"] = float(
                np.mean((pred_reward - true_reward_norm) ** 2)
            )
            out["reward_mae"] = float(
                np.mean(np.abs(pred_reward - true_reward_norm))
            )
            out["reward_true_mean"] = float(np.mean(true_reward_norm))
            out["reward_pred_mean"] = float(np.mean(pred_reward))
            rewards_norm = true_reward_norm
            true_bins = encoder.primary_bin_index(rewards_norm)
            pred_bins = encoder.decode_bin_indices(pred_reward_logits)
            twohot = encoder.encode_twohot(rewards_norm)
            out["true_reward_bins"] = true_bins
            out["pred_reward_bins"] = pred_bins
            out["reward_bin_acc"] = float(np.mean(true_bins == pred_bins))
            out["reward_bin_ce"] = encoder.cross_entropy(
                pred_reward_logits, twohot
            )

    return out


def evaluate_dynamics_on_demos(
    dynamics: EnsembleDynamics,
    demo_path: str,
    *,
    num_trajs: int = 5,
    seed: int = 0,
    record_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    traj_spans: Optional[List[Tuple[int, int]]] = None,
    fixed_trajs: bool = False,
) -> Dict[str, float]:
    """Evaluate 1-step delta and reward error on demo trajectories.

    If ``traj_spans`` is given, those (start, end) ranges are used directly.
    Else if ``fixed_trajs`` is True, trajs are sampled once from ``seed``
    (not re-randomized per epoch). Otherwise sampling uses ``seed + epoch``.

    Optionally writes true/pred delta and reward arrays under ``record_dir``.
    """
    data = load_demo_arrays(demo_path)
    trajs = split_trajectories(data)
    if traj_spans is not None:
        sampled = list(traj_spans)
    else:
        sample_seed = seed if (fixed_trajs or epoch is None) else seed + int(epoch)
        rng = np.random.default_rng(sample_seed)
        sampled = sample_trajectory_indices(trajs, num_trajs, rng)

    obs_list, act_list, next_obs_list, reward_list = [], [], [], []
    for start, end in sampled:
        obs_list.append(data["observations"][start:end])
        act_list.append(data["actions"][start:end])
        next_obs_list.append(data["next_observations"][start:end])
        reward_list.append(data["rewards"][start:end])

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    rewards = np.concatenate(reward_list, axis=0)

    result = evaluate_delta_on_transitions(
        dynamics, observations, actions, next_observations, rewards=rewards
    )

    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
        tag = f"epoch_{epoch:05d}" if epoch is not None else "eval"
        np.savez_compressed(
            os.path.join(record_dir, f"delta_compare_{tag}.npz"),
            true_delta=result["true_delta"],
            pred_delta=result["pred_delta"],
            error=result["error"],
            true_reward=result.get("true_reward"),
            pred_reward=result.get("pred_reward"),
            reward_error=result.get("reward_error"),
            true_reward_bins=result.get("true_reward_bins"),
            pred_reward_bins=result.get("pred_reward_bins"),
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            traj_spans=np.asarray(sampled, dtype=np.int64),
        )

    metrics: Dict[str, float] = {
        "eval/delta_mse": result["mse"],
        "eval/delta_mae": result["mae"],
        "eval/reward_true_mean": result.get("reward_true_mean", float("nan")),
        "eval/reward_pred_mean": result.get("reward_pred_mean", float("nan")),
    }
    if "reward_bin_acc" in result:
        metrics["eval/reward_bin_acc"] = result["reward_bin_acc"]
        metrics["eval/reward_bin_ce"] = result["reward_bin_ce"]
    else:
        metrics["eval/reward_mse"] = result.get("reward_mse", float("nan"))
        metrics["eval/reward_mae"] = result.get("reward_mae", float("nan"))
    for i, (mse_i, mae_i) in enumerate(
        zip(result["mse_per_dim"], result["mae_per_dim"])
    ):
        metrics[f"eval/delta_mse_dim{i}"] = float(mse_i)
        metrics[f"eval/delta_mae_dim{i}"] = float(mae_i)
    return metrics


def make_eval_callback(
    demo_path: str,
    *,
    num_trajs: int = 5,
    seed: int = 0,
    record_dir: Optional[str] = None,
    fixed_trajs: bool = True,
) -> Callable[[EnsembleDynamics, int], Dict[str, float]]:
    """Build a callback ``(dynamics, epoch) -> metrics`` for dynamics training.

    When ``fixed_trajs`` is True (default), the same demo trajectories are used
    every epoch so curves are comparable; checkpoint selection still uses
    holdout loss inside ``EnsembleDynamics.train``.
    """
    cached_spans: Optional[List[Tuple[int, int]]] = None
    if fixed_trajs:
        data = load_demo_arrays(demo_path)
        trajs = split_trajectories(data)
        rng = np.random.default_rng(seed)
        cached_spans = sample_trajectory_indices(trajs, num_trajs, rng)

    def callback(dynamics: EnsembleDynamics, epoch: int) -> Dict[str, float]:
        return evaluate_dynamics_on_demos(
            dynamics,
            demo_path,
            num_trajs=num_trajs,
            seed=seed,
            record_dir=record_dir,
            epoch=epoch,
            traj_spans=cached_spans,
            fixed_trajs=fixed_trajs,
        )

    return callback


def _build_dynamics_from_checkpoint(
    load_path: str,
    *,
    obs_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    n_ensemble: int,
    n_elites: int,
    weight_decay: List[float],
    task: str,
    device: str,
    reward_mode: str = "twohot",
    num_reward_bins: int = 255,
) -> EnsembleDynamics:
    from offlinerlkit.utils.reward_encoding import RewardEncodingConfig

    reward_config = RewardEncodingConfig.load(load_path)
    if reward_config.reward_mode == "gaussian_joint":
        reward_mode = "gaussian_joint"
    else:
        reward_mode = "twohot"
        num_reward_bins = reward_config.encoder.num_bins
    model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        num_ensemble=n_ensemble,
        num_elites=n_elites,
        weight_decays=weight_decay,
        reward_mode=reward_mode,
        num_reward_bins=num_reward_bins,
        device=device,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = StandardScaler()
    dynamics = EnsembleDynamics(
        model, optim, scaler, get_termination_fn(task=task), reward_config=reward_config
    )
    dynamics.load(load_path)
    return dynamics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dynamics-model delta and reward predictions to demo trajectories."
        )
    )
    parser.add_argument(
        "--demo-path",
        type=str,
        default=DEFAULT_DEMO_PATH,
        help="Path to demo HDF5 (default: mountain_car_human.hdf5).",
    )
    parser.add_argument(
        "--load-dynamics-path",
        type=str,
        required=True,
        help="Directory containing dynamics.pth and scaler.",
    )
    parser.add_argument("--task", type=str, default="mountaincar-human-v0")
    parser.add_argument("--num-trajs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="If set, save true/pred delta and reward arrays as .npz here.",
    )
    parser.add_argument("--obs-dim", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument(
        "--dynamics-hidden-dims", type=int, nargs="*", default=[200, 200, 200, 200]
    )
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument(
        "--dynamics-weight-decay",
        type=float,
        nargs="*",
        default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    dynamics = _build_dynamics_from_checkpoint(
        args.load_dynamics_path,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        n_ensemble=args.n_ensemble,
        n_elites=args.n_elites,
        weight_decay=args.dynamics_weight_decay,
        task=args.task,
        device=args.device,
    )
    metrics = evaluate_dynamics_on_demos(
        dynamics,
        args.demo_path,
        num_trajs=args.num_trajs,
        seed=args.seed,
        record_dir=args.record_dir,
    )
    print(f"Demo: {args.demo_path}")
    print(f"Model: {args.load_dynamics_path}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()

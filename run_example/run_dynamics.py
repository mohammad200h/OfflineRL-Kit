import argparse
import os
import sys
import random

import numpy as np
import torch

import h5py

from offlinerlkit.utils.d4rl_env import make_env, set_env_seed, download_dataset

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.reward_encoding import (
    RewardEncodingConfig,
    SymlogTwoHotEncoder,
    fit_reward_preprocessor,
)
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import COMBOPolicy
from wandb_utils import add_wandb_args, init_wandb, finish_wandb
from eval_dynamics import make_eval_callback


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=False)
    parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument(
        "--dynamics-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for ensemble dynamics training",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of transitions held out for dynamics validation / elite selection "
             "(capped at 1000 samples inside EnsembleDynamics.train)",
    )
    parser.add_argument(
        "--logvar-loss-coef",
        type=float,
        default=0.01,
        help="Coefficient on the Gaussian log-variance regularizer in dynamics training",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="twohot",
        choices=["twohot", "gaussian_joint"],
        help="Reward head: symlog twohot categorical (default) or legacy joint Gaussian",
    )
    parser.add_argument(
        "--num-reward-bins",
        type=int,
        default=255,
        help="Number of symlog bins for twohot reward prediction",
    )
    parser.add_argument(
        "--reward-loss-weight",
        type=float,
        default=1.0,
        help="Weight on reward CE loss (twohot mode)",
    )
    parser.add_argument(
        "--dynamics-loss-weight",
        type=float,
        default=1.0,
        help="Weight on state-delta Gaussian loss (twohot mode)",
    )
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)
    parser.add_argument(
        "--dynamics-max-epochs",
        type=int,
        default=None,
        help="Hard cap on dynamics training epochs (None = early-stop only)",
    )
    parser.add_argument(
        "--max-epochs-since-update",
        type=int,
        default=5,
        help="Early-stop patience; set very large to effectively disable early stopping",
    )
    parser.add_argument(
        "--output-model-name",
        type=str,
        default=None,
        help="Save under models/dynamics-ensemble/<seed>/<output-model-name>/ "
             "(default: <task>). Also used as wandb run name if --wandb-name is unset.",
    )

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--eval-demo-path",
        type=str,
        default=None,
        help="HDF5 demos for dynamics delta/reward eval. "
             "Default: the offline dataset for --task (D4RL / local demo path).",
    )
    parser.add_argument(
        "--eval-num-trajs",
        type=int,
        default=10,
        help="Number of demo trajectories for dynamics delta/reward eval (default: 10).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1,
        help="Run delta eval every N dynamics epochs (also once after elite selection). "
             "Set 0 to disable mid-training eval (final eval still runs if demo path exists).",
    )
    parser.add_argument(
        "--eval-record-dir",
        type=str,
        default=None,
        help="If set, save true/pred delta and reward .npz files here each eval. "
             "Default when None: <save_dir>/delta_eval",
    )
    parser.add_argument(
        "--eval-fixed-trajs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the same demo trajectories every eval (default: on). "
             "Disable with --no-eval-fixed-trajs for random resampling.",
    )
    parser.add_argument(
        "--no-eval-dynamics",
        action="store_true",
        help="Disable trajectory delta/reward evaluation during dynamics training.",
    )

    add_wandb_args(parser)
    return parser.parse_args()

def load_neorl_dataset(env, data_type, traj_num=1000):
    train_data, _ = env.get_dataset(data_type=data_type, train_num=traj_num, need_val=False)
    dataset = {}
    dataset["observations"] = train_data["obs"]
    dataset["actions"] = train_data["action"]
    dataset["next_observations"] = train_data["next_obs"]
    dataset["rewards"] = train_data["reward"]
    dataset["terminals"] = train_data["done"]
    return dataset


def resolve_eval_demo_path(args, env, *, is_neorl: bool) -> str | None:
    """Pick the HDF5 used for mid-training delta eval.

    Prefer an explicit --eval-demo-path. Otherwise use the same offline dataset
    as training (so Walker2D / Hopper / etc. do not fall back to MountainCar demos).
    """
    if args.eval_demo_path:
        return args.eval_demo_path
    if is_neorl:
        return None
    dataset_url = getattr(env, "_dataset_url", None)
    if not dataset_url:
        return None
    return download_dataset(dataset_url)


def demo_matches_env(demo_path: str, obs_dim: int, action_dim: int) -> bool:
    """Return True if demo HDF5 obs/action widths match the training env."""
    with h5py.File(demo_path, "r") as f:
        if "observations" not in f or "actions" not in f:
            return False
        demo_obs = int(np.prod(f["observations"].shape[1:])) if f["observations"].ndim > 1 else 1
        act = f["actions"]
        demo_act = 1 if act.ndim == 1 else int(np.prod(act.shape[1:]))
        return demo_obs == obs_dim and demo_act == action_dim


def train(args=get_args()):
    is_neorl = args.task.split('-')[1] == 'v3'

    # create env and dataset
    if is_neorl:
        import neorl
        task, version, data_type = tuple(args.task.split("-"))
        env = neorl.make(task+'-'+version)
        dataset = load_neorl_dataset(env, data_type)
    else:
        env = make_env(args.task)
        dataset = qlearning_dataset(env)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    if env.observation_space.shape is not None:
        args.obs_shape = env.observation_space.shape
    else:
        args.obs_shape = dataset["observations"].shape[1:]
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if is_neorl:
        env.seed(args.seed)
    else:
        set_env_seed(env, args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        reward_mode=args.reward_mode,
        num_reward_bins=args.num_reward_bins,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    reward_config = RewardEncodingConfig(
        reward_mode=args.reward_mode,
        reward_loss_weight=args.reward_loss_weight,
        dynamics_loss_weight=args.dynamics_loss_weight,
        preprocessor=fit_reward_preprocessor(dataset["rewards"], args.task),
        encoder=SymlogTwoHotEncoder(num_bins=args.num_reward_bins),
    )
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        reward_config=reward_config,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)

    model_name = args.output_model_name or args.task
    wandb_name = args.wandb_name or args.output_model_name

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    init_wandb(
        args.track,
        args.project,
        wandb_name,
        vars(args),
        log_dirs=log_dirs,
    )

    save_dir = os.path.join('./models/dynamics-ensemble/', str(args.seed), model_name)
    os.makedirs(save_dir, exist_ok=True)

    eval_callback = None
    eval_freq = args.eval_freq
    if not args.no_eval_dynamics:
        eval_demo_path = resolve_eval_demo_path(args, env, is_neorl=is_neorl)
        obs_dim = int(np.prod(args.obs_shape))
        if eval_demo_path is None or not os.path.isfile(eval_demo_path):
            logger.log(
                f"Warning: eval demo not found"
                f"{'' if eval_demo_path is None else f' at {eval_demo_path}'}; "
                "skipping dynamics delta eval."
            )
        elif not demo_matches_env(eval_demo_path, obs_dim, int(args.action_dim)):
            logger.log(
                f"Warning: eval demo at {eval_demo_path} does not match "
                f"env dims obs={obs_dim} act={args.action_dim}; "
                "skipping dynamics delta eval. Pass --eval-demo-path for this task."
            )
        else:
            logger.log(f"Dynamics delta eval demos: {eval_demo_path}")
            record_dir = args.eval_record_dir
            if record_dir is None:
                record_dir = os.path.join(save_dir, "delta_eval")
            eval_callback = make_eval_callback(
                eval_demo_path,
                num_trajs=args.eval_num_trajs,
                seed=args.seed,
                record_dir=record_dir,
                fixed_trajs=args.eval_fixed_trajs,
            )
            # eval_freq=0: skip mid-training; train() still runs a final eval.
            if eval_freq == 0:
                eval_freq = 10**9

    train_data = real_buffer.sample_all()
    train_data["task"] = args.task
    dynamics.train(
        train_data,
        logger,
        max_epochs=args.dynamics_max_epochs,
        max_epochs_since_update=args.max_epochs_since_update,
        batch_size=args.dynamics_batch_size,
        holdout_ratio=args.holdout_ratio,
        logvar_loss_coef=args.logvar_loss_coef,
        eval_callback=eval_callback,
        eval_freq=eval_freq,
    )
    dynamics.save(save_dir)
    finish_wandb(args.track)


if __name__ == "__main__":
    train()

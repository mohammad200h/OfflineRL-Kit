try:
    import wandb
except ImportError:
    wandb = None


def add_wandb_args(parser):
    parser.add_argument(
        '--track',
        action='store_true',
        help='Log training metrics natively to Weights & Biases',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='offlin RL',
        help='WandB Cloud project name',
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='WandB run name (default: <algo-name>_<task>)',
    )
    return parser


def init_wandb(
    track: bool,
    project_name: str,
    wandb_name: str | None,
    config: dict,
    log_dirs: str | None = None,
) -> None:
    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        init_kwargs = {
            'project': project_name,
            'name': wandb_name or (
                f'{config.get("algo_name", "run")}_{config["task"]}'
            ),
            'config': config,
            'sync_tensorboard': True,
            'monitor_gym': True,
            'save_code': True,
        }
        if log_dirs is not None:
            init_kwargs['dir'] = log_dirs
        wandb.init(**init_kwargs)


def finish_wandb(track: bool) -> None:
    if track:
        wandb.finish()

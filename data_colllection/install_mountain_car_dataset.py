#!/usr/bin/env python3
"""Install MountainCar demos into the OfflineRL-Kit / D4RL dataset cache.

``run_dynamics.py`` loads datasets via ``make_env`` → ``env.get_dataset()``,
which reads::

    $D4RL_DATASET_DIR/mountain_car_human.hdf5
    # default: ~/.d4rl/datasets/mountain_car_human.hdf5

Dropbox demos land in ``data_colllection/demostrations/``. This script
symlinks (or copies) that file into the expected cache path so
``--task mountaincar-human-v0`` works without editing ``run_dynamics.py``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "demostrations" / "mountain_car_human.hdf5"
DATASET_NAME = "mountain_car_human.hdf5"


def default_dataset_dir() -> Path:
    return Path(
        os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets"))
    ).expanduser().resolve()


def install(source: Path, dest_dir: Path, *, copy: bool, force: bool) -> Path:
    if not source.is_file():
        raise SystemExit(
            f"Source dataset not found: {source}\n"
            "Download first:\n"
            "  python3 data_colllection/download_from_dropbox.py"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / DATASET_NAME

    if dest.exists() or dest.is_symlink():
        if not force:
            # Already points at the same file
            try:
                if dest.resolve() == source.resolve():
                    print(f"already installed: {dest} -> {source}")
                    return dest
            except FileNotFoundError:
                pass
            raise SystemExit(
                f"Destination already exists: {dest}\n"
                "Re-run with --force to replace it."
            )
        if dest.is_dir() and not dest.is_symlink():
            raise SystemExit(f"Refusing to replace directory: {dest}")
        dest.unlink()

    if copy:
        shutil.copy2(source, dest)
        print(f"copied: {source} -> {dest}")
    else:
        dest.symlink_to(source.resolve())
        print(f"linked: {dest} -> {source.resolve()}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Demo HDF5 path (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "D4RL dataset cache directory "
            f"(default: $D4RL_DATASET_DIR or {default_dataset_dir()})"
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the file instead of creating a symlink",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file/symlink at the destination",
    )
    args = parser.parse_args()

    dest_dir = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir is not None
        else default_dataset_dir()
    )
    install(args.source.resolve(), dest_dir, copy=args.copy, force=args.force)
    print(
        "Ready for: python3 run_example/run_dynamics.py "
        "--task mountaincar-human-v0 --seed 1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

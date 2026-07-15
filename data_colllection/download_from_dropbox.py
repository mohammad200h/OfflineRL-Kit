#!/usr/bin/env python3
"""Download demonstration files from Dropbox into data_colllection/demostrations.

Only downloads a file when it is missing locally or the Dropbox content hash
differs from the local file (i.e. content has changed).

Reads DROPBOX_TOKEN from docker/.env (or the environment). Falls back to
DROPBOX_API only if it looks like an access token (starts with ``sl.``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FileMetadata

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = SCRIPT_DIR / "demostrations"
DEFAULT_ENV = SCRIPT_DIR.parent / "docker" / ".env"
DROPBOX_ROOT = "/OfflineRL-Kit/demostrations"
BLOCK_SIZE = 4 * 1024 * 1024  # Dropbox content-hash block size


def _parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_token(env_path: Path) -> str:
    env = _parse_env_file(env_path)
    for key in ("DROPBOX_TOKEN", "DROPBOX_API"):
        token = os.environ.get(key, "").strip() or env.get(key, "").strip()
        if not token:
            continue
        if key == "DROPBOX_API" and not token.startswith("sl."):
            continue  # app key, not an access token
        return token
    raise SystemExit(
        f"DROPBOX_TOKEN not set in environment or {env_path}. "
        "Put your Generated access token there."
    )


def dropbox_content_hash(path: Path) -> str:
    """Compute Dropbox content_hash for a local file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            hasher.update(hashlib.sha256(block).digest())
    return hasher.hexdigest()


def iter_remote_files(dbx: dropbox.Dropbox, remote_dir: str):
    try:
        result = dbx.files_list_folder(remote_dir, recursive=True)
    except ApiError as err:
        if err.error.is_path() and err.error.get_path().is_not_found():
            return
        raise
    entries = list(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)
    for entry in entries:
        if isinstance(entry, FileMetadata):
            yield entry


def download_file(dbx: dropbox.Dropbox, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    metadata, response = dbx.files_download(remote_path)
    local_path.write_bytes(response.content)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV,
        help=f"Path to .env with DROPBOX_TOKEN (default: {DEFAULT_ENV})",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=LOCAL_DIR,
        help=f"Local demonstrations directory (default: {LOCAL_DIR})",
    )
    parser.add_argument(
        "--remote-dir",
        default=DROPBOX_ROOT,
        help=f"Dropbox source folder (default: {DROPBOX_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download all files even when content hashes match",
    )
    args = parser.parse_args()

    local_dir = args.local_dir.resolve()
    remote_dir = args.remote_dir.rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)

    token = load_token(args.env)
    dbx = dropbox.Dropbox(token)
    try:
        dbx.users_get_current_account()
    except AuthError:
        print(
            "Dropbox auth failed: check DROPBOX_TOKEN in docker/.env "
            "(Generated access token from the App Console).",
            file=sys.stderr,
        )
        return 1

    remote_files = list(iter_remote_files(dbx, remote_dir))
    if not remote_files:
        print(f"No files under Dropbox path {remote_dir}")
        return 0

    downloaded = skipped = 0
    for meta in remote_files:
        # meta.path_display like "/OfflineRL-Kit/demostrations/foo.hdf5"
        rel = meta.path_display[len(remote_dir) :].lstrip("/")
        if not rel:
            continue
        local_path = local_dir / rel
        if (
            not args.force
            and local_path.is_file()
            and meta.content_hash
            and dropbox_content_hash(local_path) == meta.content_hash
        ):
            print(f"skip (unchanged): {rel}")
            skipped += 1
            continue
        reason = "new" if not local_path.exists() else "changed"
        print(f"download ({reason}): {meta.path_display} -> {local_path}")
        download_file(dbx, meta.path_display, local_path)
        downloaded += 1

    print(f"Done. downloaded={downloaded} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

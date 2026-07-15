#!/usr/bin/env python3
"""Upload files from data_colllection/demostrations to Dropbox.

Only uploads a file when it is missing remotely or the Dropbox content hash
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
from dropbox.files import FileMetadata, WriteMode

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


def remote_content_hash(dbx: dropbox.Dropbox, remote_path: str) -> str | None:
    try:
        meta = dbx.files_get_metadata(remote_path)
    except ApiError as err:
        if err.error.is_path() and err.error.get_path().is_not_found():
            return None
        raise
    if isinstance(meta, FileMetadata):
        return meta.content_hash
    return None


def upload_file(dbx: dropbox.Dropbox, local_file: Path, remote_path: str) -> None:
    size = local_file.stat().st_size
    # Simple upload for files under 150 MB; session upload otherwise.
    if size <= 150 * 1024 * 1024:
        with local_file.open("rb") as f:
            dbx.files_upload(f.read(), remote_path, mode=WriteMode.overwrite, mute=True)
        return

    chunk = 8 * 1024 * 1024
    with local_file.open("rb") as f:
        data = f.read(chunk)
        session = dbx.files_upload_session_start(data)
        offset = len(data)
        cursor = dropbox.files.UploadSessionCursor(session_id=session.session_id, offset=offset)
        commit = dropbox.files.CommitInfo(path=remote_path, mode=WriteMode.overwrite)
        while True:
            data = f.read(chunk)
            if not data:
                break
            if offset + len(data) < size:
                dbx.files_upload_session_append_v2(data, cursor)
                offset += len(data)
                cursor.offset = offset
            else:
                dbx.files_upload_session_finish(data, cursor, commit)
                return
        dbx.files_upload_session_finish(b"", cursor, commit)


def iter_local_files(local_dir: Path):
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


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
        help=f"Dropbox destination folder (default: {DROPBOX_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload all files even when content hashes match",
    )
    args = parser.parse_args()

    local_dir = args.local_dir.resolve()
    remote_dir = args.remote_dir.rstrip("/")

    if not local_dir.is_dir():
        raise SystemExit(f"Local directory not found: {local_dir}")

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

    files = list(iter_local_files(local_dir))
    if not files:
        print(f"No files under {local_dir}")
        return 0

    uploaded = skipped = 0
    for local_file in files:
        rel = local_file.relative_to(local_dir).as_posix()
        remote = f"{remote_dir}/{rel}"
        local_hash = dropbox_content_hash(local_file)
        remote_hash = None if args.force else remote_content_hash(dbx, remote)
        if remote_hash is not None and remote_hash == local_hash:
            print(f"skip (unchanged): {rel}")
            skipped += 1
            continue
        reason = "new" if remote_hash is None else "changed"
        print(f"upload ({reason}): {rel} -> {remote}")
        upload_file(dbx, local_file, remote)
        uploaded += 1

    print(f"Done. uploaded={uploaded} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

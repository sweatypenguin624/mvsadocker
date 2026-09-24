#!/usr/bin/env python3
"""Download a single video from Google Drive by its COMPLETE Drive path.

Reusable standalone (CLI) and as a function (download_one) called by
batch_pipeline.py. Never touches queue/manifest.csv itself -- callers decide
what to do with the returned status.

Usage:
    python scripts/download_video.py --drive-path "Data Corp /.../08.00.00-09.00.00[R][0@0][0].dav" \\
        --dest work/downloads/peds10_cam1_20260728_0800.dav
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from batch_utils import PROJECT_ROOT, load_pipeline_config, resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_video")


class DownloadError(RuntimeError):
    pass


def download_one(
    drive_path: str,
    dest: Path,
    cfg: dict,
    log_write=None,
) -> Path:
    """Download exactly one file, identified by its full Drive path, to
    ``dest``. Retries up to rclone.max_retries times on failure. Raises
    DownloadError (never silently returns a partial/missing file) if every
    attempt fails.
    """
    rclone_cfg = cfg["rclone"]
    rclone_bin = resolve(rclone_cfg["binary"])
    rclone_conf = resolve(rclone_cfg["config"])
    remote = rclone_cfg["remote"]
    max_retries = int(rclone_cfg.get("max_retries", 3))
    backoff = int(rclone_cfg.get("retry_backoff_seconds", 10))
    timeout = int(rclone_cfg.get("timeout_seconds", 3600))

    if not rclone_bin.exists():
        raise FileNotFoundError(f"rclone binary not found: {rclone_bin}")
    if not rclone_conf.exists():
        raise FileNotFoundError(f"rclone config not found: {rclone_conf}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_source = f"{remote}{drive_path}"

    def emit(msg: str) -> None:
        logger.info(msg)
        if log_write:
            log_write(msg)

    if dest.exists() and dest.stat().st_size > 0:
        emit(f"Already downloaded, skipping: {dest}")
        return dest

    last_err = ""
    for attempt in range(1, max_retries + 1):
        emit(f"Download attempt {attempt}/{max_retries}: {remote_source} -> {dest}")
        cmd = [
            str(rclone_bin),
            "copyto",
            "--config",
            str(rclone_conf),
            remote_source,
            str(dest),
        ]
        if rclone_cfg.get("shared_with_me", True):
            cmd.append("--drive-shared-with-me")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_err = f"timed out after {timeout}s"
            emit(f"Attempt {attempt} {last_err}")
        else:
            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                emit(f"Downloaded {dest.stat().st_size} bytes -> {dest}")
                return dest
            last_err = result.stderr.strip() or f"exit code {result.returncode}"
            emit(f"Attempt {attempt} failed: {last_err}")
            if dest.exists():
                dest.unlink()

        if attempt < max_retries:
            time.sleep(backoff)

    raise DownloadError(f"Failed to download after {max_retries} attempts: {drive_path} ({last_err})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-path", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.pipeline_config)
    try:
        download_one(args.drive_path, args.dest, cfg)
    except (DownloadError, FileNotFoundError) as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Remux a downloaded .dav file into a .mp4 usable by Ultralytics/OpenCV.

Thin wrapper around the existing engine's scripts/video_utils.remux_dav_to_mp4
(stream copy, no re-encode -- reused as-is, not reimplemented). Reusable
standalone (CLI) and as a function (convert_one) called by batch_pipeline.py.

Usage:
    python scripts/convert_video.py --dav work/downloads/x.dav --mp4 work/converted/x.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from batch_utils import PROJECT_ROOT, load_pipeline_config, resolve
from video_utils import remux_dav_to_mp4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_video")


def convert_one(dav_path: Path, mp4_path: Path, ffmpeg_bin: Path, log_write=None) -> Path:
    def emit(msg: str) -> None:
        logger.info(msg)
        if log_write:
            log_write(msg)

    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        emit(f"Already converted, skipping: {mp4_path}")
        return mp4_path

    emit(f"Remuxing {dav_path} -> {mp4_path}")
    remux_dav_to_mp4(dav_path, mp4_path, ffmpeg_bin)
    emit(f"Converted: {mp4_path} ({mp4_path.stat().st_size} bytes)")
    return mp4_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dav", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.pipeline_config)
    ffmpeg_bin = resolve(cfg["tools"]["ffmpeg"])
    try:
        convert_one(args.dav, args.mp4, ffmpeg_bin)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

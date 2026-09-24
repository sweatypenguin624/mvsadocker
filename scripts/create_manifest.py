#!/usr/bin/env python3
"""Select 10-20 representative videos from queue/candidates.csv and write
queue/manifest.csv -- the authoritative list of exactly which videos this
validation batch will download and process, each keyed by its COMPLETE
Drive path plus a derived run_name.

Selection strategy: round-robin across cameras (maximizes site/camera
diversity within a small budget), each camera contributing its available
hour that is earliest in config.selection.preferred_hours, until
target_video_count is reached or candidates are exhausted. Deterministic --
no randomness -- so re-running with the same candidates.csv reproduces the
same manifest.

Usage:
    python scripts/create_manifest.py [--pipeline-config config/pipeline.yaml]
        [--candidates queue/candidates.csv] [--out queue/manifest.csv]
        [--count N] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from batch_utils import (
    PROJECT_ROOT,
    derive_run_name,
    load_pipeline_config,
    now_iso,
    read_candidates,
    write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_manifest")


def select_videos(candidates: list, target_count: int, preferred_hours: list, min_size_mb: int) -> list:
    min_size_bytes = min_size_mb * 1024 * 1024

    by_camera = defaultdict(list)
    for row in candidates:
        try:
            size = int(row.get("size_bytes") or 0)
        except ValueError:
            size = 0
        if size < min_size_bytes:
            logger.warning("Skipping undersized candidate (%d bytes < %d MB floor): %s", size, min_size_mb, row["drive_path"])
            continue
        by_camera[(row["site"], row["camera"])].append(row)

    def hour_rank(row: dict) -> int:
        if not row.get("hour_start"):
            return len(preferred_hours) + 1
        hh = int(row["hour_start"].split(":")[0])
        return preferred_hours.index(hh) if hh in preferred_hours else len(preferred_hours)

    for key in by_camera:
        by_camera[key].sort(key=hour_rank)

    cameras = sorted(by_camera.keys())
    if not cameras:
        return []

    selected = []
    cursor = {cam: 0 for cam in cameras}
    while len(selected) < target_count:
        made_progress = False
        for cam in cameras:
            if len(selected) >= target_count:
                break
            idx = cursor[cam]
            if idx < len(by_camera[cam]):
                selected.append(by_camera[cam][idx])
                cursor[cam] = idx + 1
                made_progress = True
        if not made_progress:
            break  # exhausted every camera's candidates before hitting target_count

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "queue" / "candidates.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "queue" / "manifest.csv")
    parser.add_argument("--count", type=int, default=None, help="Override selection.target_video_count")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.csv (loses any in-progress status)")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        logger.error(
            "%s already exists. Re-generating would discard in-progress download/processing "
            "status. Pass --force to regenerate anyway (progress will be lost).",
            args.out,
        )
        return 1

    cfg = load_pipeline_config(args.pipeline_config)
    sel_cfg = cfg.get("selection", {})
    target_count = args.count if args.count is not None else int(sel_cfg.get("target_video_count", 15))
    if not (10 <= target_count <= 20):
        logger.warning("target_video_count=%d is outside the required 10-20 validation-batch range", target_count)

    candidates = read_candidates(args.candidates)
    if not candidates:
        logger.error("%s is empty or missing -- run discover_videos.py first", args.candidates)
        return 1

    preferred_hours = list(sel_cfg.get("preferred_hours", []))
    min_size_mb = int(sel_cfg.get("min_size_mb", 0))

    selected = select_videos(candidates, target_count, preferred_hours, min_size_mb)
    if not selected:
        logger.error("No candidates survived selection (check min_size_mb / candidates.csv contents)")
        return 1

    seen_run_names = set()
    rows = []
    for row in selected:
        run_name = derive_run_name(row["site"], row["camera"], row["date"], row["hour_start"])
        base_run_name = run_name
        suffix = 2
        while run_name in seen_run_names:
            run_name = f"{base_run_name}_{suffix}"
            suffix += 1
        seen_run_names.add(run_name)

        rows.append(
            {
                "run_name": run_name,
                "drive_path": row["drive_path"],
                "site": row["site"],
                "camera": row["camera"],
                "date": row["date"],
                "hour_start": row["hour_start"],
                "hour_end": row["hour_end"],
                "size_bytes": row["size_bytes"],
                "status": "pending",
                "retries": "0",
                "download_path": "",
                "converted_path": "",
                "results_dir": "",
                "error": "",
                "added_at": now_iso(),
                "downloaded_at": "",
                "converted_at": "",
                "processed_at": "",
            }
        )

    write_manifest(args.out, rows)
    logger.info("Selected %d videos across %d cameras -> %s", len(rows), len({(r["site"], r["camera"]) for r in rows}), args.out)
    for r in rows:
        logger.info("  %s  <-  %s", r["run_name"], r["drive_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

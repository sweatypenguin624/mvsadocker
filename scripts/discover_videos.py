#!/usr/bin/env python3
"""Discover candidate video files from Google Drive without downloading them.

Lists (metadata only -- Path/Size/ModTime, never file contents) every
candidate video file under config/pipeline.yaml's drive.base_path, and
writes one row per file to queue/candidates.csv, always keyed by its
COMPLETE Drive path (never by filename alone -- the same filename recurs
under every camera/date).

Traverses site -> camera -> files as three separate rclone calls per camera,
rather than one recursive call over the whole base_path tree. A single
recursive lsjson over this tree was observed to silently under-list files
(84 .dav files present, only 60 returned) -- almost certainly a Google
Drive API eventual-consistency/pagination quirk on a deeply-nested "shared
with me" tree, not something rclone reported as an error. Scoping each
listing call to one camera folder (~12-30 files) was consistently complete
across repeated manual checks, so that's the unit of work here.

Recognizes two filename conventions seen in this dataset:
  - "HH.MM.SS-HH.MM.SS[R][0@0][0].dav"          (hour-range, most cameras)
  - "YYYYMMDD_HHMMSS_tpNNNNN.mp4"                 (single start-time, e.g.
                                                    Kamaripet Cam 5's DVR)

Usage:
    python scripts/discover_videos.py [--pipeline-config config/pipeline.yaml] [--out queue/candidates.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from batch_utils import (
    PROJECT_ROOT,
    load_pipeline_config,
    now_iso,
    resolve,
    write_candidates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("discover_videos")

RANGE_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{2})-(\d{2}\.\d{2}\.\d{2})")
TIMESTAMP_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")

DEFAULT_EXTENSIONS = (".dav", ".mp4")


def _rclone_base_cmd(cfg: dict) -> tuple:
    rclone_cfg = cfg["rclone"]
    rclone_bin = resolve(rclone_cfg["binary"])
    rclone_conf = resolve(rclone_cfg["config"])
    if not rclone_bin.exists():
        raise FileNotFoundError(f"rclone binary not found: {rclone_bin}")
    if not rclone_conf.exists():
        raise FileNotFoundError(f"rclone config not found: {rclone_conf}")
    extra = ["--drive-shared-with-me"] if rclone_cfg.get("shared_with_me", True) else []
    return str(rclone_bin), str(rclone_conf), extra, rclone_cfg.get("timeout_seconds", 3600)


def _list_dirs(rclone_bin: str, rclone_conf: str, extra: list, timeout: int, path: str) -> list:
    cmd = [rclone_bin, "lsf", "--config", rclone_conf, path, "--dirs-only"] + extra
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"rclone lsf --dirs-only failed for {path} (exit {result.returncode}):\n{result.stderr}")
    return [line.rstrip("/") for line in result.stdout.splitlines() if line.strip()]


def _list_files_recursive(rclone_bin: str, rclone_conf: str, extra: list, timeout: int, path: str) -> list:
    cmd = [rclone_bin, "lsjson", "--config", rclone_conf, path, "--recursive", "--files-only"] + extra
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed for {path} (exit {result.returncode}):\n{result.stderr}")
    return json.loads(result.stdout)


def discover(cfg: dict) -> list:
    rclone_bin, rclone_conf, extra, timeout = _rclone_base_cmd(cfg)
    remote = cfg["rclone"]["remote"]
    base_path = cfg["drive"]["base_path"]
    extensions = tuple(cfg["drive"].get("video_extensions", DEFAULT_EXTENSIONS))
    base_remote = f"{remote}{base_path}"

    # Optional, case-insensitive regex on the camera folder name -- e.g. the
    # CVC vehicle-count survey pairs most cameras with a redundant "Backup"
    # folder recording the same view a second time, which would otherwise
    # double-count camera diversity in create_manifest.py's round-robin
    # selection. Unset (the default) preserves the original behavior for the
    # Peds pipeline, which has no such folders.
    exclude_pattern = cfg["drive"].get("exclude_camera_pattern")
    exclude_re = re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None

    sites = _list_dirs(rclone_bin, rclone_conf, extra, timeout, base_remote)
    logger.info("Found %d site folder(s) under base_path", len(sites))

    all_rows = []
    for site in sites:
        site_remote = f"{base_remote}/{site}"
        try:
            cameras = _list_dirs(rclone_bin, rclone_conf, extra, timeout, site_remote)
        except RuntimeError as e:
            logger.warning("Could not list cameras under site %r: %s", site, e)
            continue

        if exclude_re is not None:
            kept = [c for c in cameras if not exclude_re.search(c)]
            skipped = [c for c in cameras if exclude_re.search(c)]
            if skipped:
                logger.info("Site %r: excluding %d camera folder(s) matching %r: %s", site, len(skipped), exclude_pattern, skipped)
            cameras = kept

        logger.info("Site %r: %d camera folder(s)", site, len(cameras))

        for camera in cameras:
            camera_remote = f"{site_remote}/{camera}"
            try:
                entries = _list_files_recursive(rclone_bin, rclone_conf, extra, timeout, camera_remote)
            except RuntimeError as e:
                logger.warning("Could not list files under %r / %r: %s", site, camera, e)
                continue

            video_entries = [e for e in entries if e.get("Path", "").lower().endswith(extensions)]
            logger.info("  %r / %r: %d video file(s)", site, camera, len(video_entries))
            if not video_entries and entries:
                logger.warning(
                    "  %r / %r has %d non-matching file(s) (extensions=%s) -- e.g. %s",
                    site, camera, len(entries), extensions, entries[0].get("Path"),
                )

            for e in video_entries:
                all_rows.append(parse_entry(e, base_path, site, camera))

    return all_rows


def parse_entry(entry: dict, base_path: str, site: str, camera: str) -> dict:
    rel_path = entry["Path"]  # relative to the camera folder, e.g. "2026-07-28/08.00.00-...dav"
    parts = rel_path.split("/")
    filename = parts[-1]
    folder_date = parts[0] if len(parts) >= 2 else ""

    date = folder_date
    hour_start = hour_end = ""

    m = RANGE_RE.match(filename)
    if m:
        hour_start = m.group(1).replace(".", ":")
        hour_end = m.group(2).replace(".", ":")
    else:
        m = TIMESTAMP_RE.match(filename)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            filename_date = f"{y}-{mo}-{d}"
            if folder_date and folder_date != filename_date:
                logger.warning(
                    "Filename date %s disagrees with folder date %s for %s -- using folder date",
                    filename_date, folder_date, filename,
                )
            else:
                date = filename_date
            hour_start = f"{hh}:{mm}:{ss}"
            # No declared end time for this convention (short, variable-length
            # DVR segments) -- left blank rather than guessed.
        else:
            logger.warning("Could not parse a start time from filename: %s", filename)

    drive_path = f"{base_path}/{site}/{camera}/{rel_path}"

    return {
        "drive_path": drive_path,
        "site": site,
        "camera": camera,
        "date": date,
        "hour_start": hour_start,
        "hour_end": hour_end,
        "filename": filename,
        "size_bytes": str(entry.get("Size", "")),
        "discovered_at": now_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "queue" / "candidates.csv")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.pipeline_config)
    rows = discover(cfg)
    rows.sort(key=lambda r: (r["site"], r["camera"], r["date"], r["hour_start"]))

    write_candidates(args.out, rows)
    logger.info("Wrote %d candidates to %s", len(rows), args.out)

    by_camera = {}
    for r in rows:
        by_camera[(r["site"], r["camera"])] = by_camera.get((r["site"], r["camera"]), 0) + 1
    for (site, camera), n in sorted(by_camera.items()):
        logger.info("  %s / %s: %d", site, camera, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())

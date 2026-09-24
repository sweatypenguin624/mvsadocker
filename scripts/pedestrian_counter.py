#!/usr/bin/env python3
"""Run the existing detection engine (YOLO11x + ByteTrack + pedestrian ROI +
counting, from config_loader.py/pipeline.py/detector.py/tracker.py/roi.py --
none of which are modified here) on ONE converted video, and print a JSON
summary to stdout.

Always invoked as its own process (never imported into a long-lived batch
loop) -- scripts/utils.py::setup_logger only attaches log handlers once per
process, so running N videos in one process would silently pile every
video's log lines into the first video's run.log. A fresh subprocess per
video keeps each results/<run_name>/logs/run.log self-contained and lets one
video's crash never take down the rest of the batch.

Each physical camera has its own field of view, so a single hardcoded ROI
polygon (config/config.yaml's roi: section) is only valid for the one camera
it was drawn against. Per-camera polygons, each recorded against its own
image's actual pixel dimensions (see scripts/roieditor.py), live in
config/roi/roi_by_camera.yaml keyed by camera_key
(slug(site)+"__"+slug(camera), e.g. "peds_15_koppikkar_road__cam_2"). When
--camera-key is passed, that camera MUST have an entry there -- there is no
silent fallback to config.yaml's default ROI, because that default is
tuned for a different camera and applying it elsewhere is exactly the bug
that motivated this per-camera lookup in the first place.

Usage:
    python scripts/pedestrian_counter.py --engine-config config/config.yaml \\
        --run-name peds10_cam1_20260728_0800 --video work/converted/x.mp4 \\
        --start-datetime "2026-07-28 08:00:00" \\
        --camera-key peds_10_kle_technological_university__cam_1 [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import yaml

from config_loader import ROIConfig, VideoConfig, load_config
from pipeline import Pipeline
from video_utils import probe_video


def load_camera_roi(roi_config_path: Path, camera_key: str) -> ROIConfig:
    if not roi_config_path.exists():
        raise FileNotFoundError(
            f"{roi_config_path} not found -- draw this camera's ROI first with scripts/roieditor.py "
            f"(--camera-key {camera_key})"
        )
    with open(roi_config_path, "r", encoding="utf-8") as f:
        all_rois = yaml.safe_load(f) or {}
    entry = all_rois.get(camera_key)
    if entry is None:
        raise KeyError(
            f"No ROI entry for camera_key='{camera_key}' in {roi_config_path} -- draw it first with "
            f"scripts/roieditor.py --camera-key {camera_key} --image config/roi/frames/{camera_key}.jpg"
        )
    return ROIConfig(
        polygon=[[float(x), float(y)] for x, y in entry["polygon"]],
        reference_width=int(entry["reference_width"]),
        reference_height=int(entry["reference_height"]),
    )


def load_camera_movement_axis(roi_config_path: Path, camera_key: str) -> Optional[dict]:
    """Look up this camera's "movement_axis" calibration (see
    scripts/movement_direction.py) from the same per-camera file as its
    ROI. Unlike the ROI itself, this is optional -- a camera with no
    movement_axis entry simply runs without direction classification,
    exactly as if this function were never called.
    """
    if not roi_config_path.exists():
        return None
    with open(roi_config_path, "r", encoding="utf-8") as f:
        all_rois = yaml.safe_load(f) or {}
    entry = all_rois.get(camera_key)
    if entry is None:
        return None
    axis = entry.get("movement_axis")
    if axis is None:
        return None
    # near_point/far_point are calibrated in the same reference_width x
    # reference_height space as this camera's ROI polygon -- carry those
    # over from the camera entry rather than requiring them duplicated
    # inside movement_axis itself.
    return {
        "near_point": axis["near_point"],
        "far_point": axis["far_point"],
        "reference_width": int(entry["reference_width"]),
        "reference_height": int(entry["reference_height"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-datetime", required=True, help="'%%Y-%%m-%%d %%H:%%M:%%S'")
    parser.add_argument("--camera-key", default=None, help="Look up a per-camera ROI override; omit to use config.yaml's default ROI as-is")
    parser.add_argument("--roi-config", type=Path, default=PROJECT_ROOT / "config" / "roi" / "roi_by_camera.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.engine_config)

    video_path = args.video.resolve()
    if not video_path.exists():
        print(json.dumps({"error": f"video not found: {video_path}"}))
        return 1

    # Probe the actual fps of THIS video rather than reusing config.yaml's
    # fps (tuned for a single reference video) -- different cameras in the
    # batch may genuinely differ, and get-it-wrong here silently skews every
    # downstream frame-index -> wall-clock-timestamp computation (hourly
    # bucketing, track first/last-seen times).
    ffprobe_bin = base_cfg.resolve(base_cfg.tools.ffprobe)
    probed = probe_video(video_path, ffprobe_bin)
    fps = probed.fps if probed.fps and probed.fps > 0 else base_cfg.video.fps

    video_cfg = VideoConfig(path=str(video_path), fps=fps, start_datetime=args.start_datetime)
    cfg = replace(base_cfg, video=video_cfg)

    if args.camera_key:
        roi_cfg = load_camera_roi(args.roi_config, args.camera_key)
        movement_axis = load_camera_movement_axis(args.roi_config, args.camera_key)
        cfg = replace(cfg, roi=roi_cfg, movement_axis=movement_axis)

    pipeline = Pipeline(cfg, run_name=args.run_name, resume=args.resume)
    summary = pipeline.run_detect()

    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

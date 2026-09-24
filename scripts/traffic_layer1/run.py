#!/usr/bin/env python3
"""Layer 1 CLI: broad-class detection + ByteTrack tracking + ROI counting +
run summary on ONE video (see the Layer 1 spec for the full class list --
Car, 2-Wheeler, Auto, Bus, Pedestrian, Emergency Vehicle, Cycle, Cycle
Rickshaw, Animal/Hand-drawn Cart, Goods Vehicle, Tractor, Others).

Usage:
    python scripts/traffic_layer1/run.py \\
        --config config/layer1_config.yaml \\
        --input videos/actual_test/08.00.00-09.00.00.mp4 \\
        --output results/layer1_run1 \\
        --start-datetime "2026-07-28 08:00:00" \\
        --camera-key peds_10_kle_technological_university__cam_1 \\
        --debug --save-crops

Run from the project root so relative paths in the config resolve:
    cd ~/mvsa && source env/bin/activate
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from video_utils import probe_video  # noqa: E402

from config import (  # noqa: E402
    AggregationConfig,
    ConfidenceConfig,
    Layer1ConfigError,
    ROIConfig,
    VideoConfig,
    load_layer1_config,
)
from layer1_pipeline import Layer1Pipeline  # noqa: E402


def load_camera_roi(roi_config_path: Path, camera_key: str, roi_by_camera_path: str) -> ROIConfig:
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
        roi_by_camera_path=roi_by_camera_path,
    )


def load_camera_movement_axis(roi_config_path: Path, camera_key: str) -> Optional[dict]:
    """Same lookup as scripts/pedestrian_counter.py::load_camera_movement_axis
    -- optional per camera, unlike the ROI itself."""
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
    return {
        "near_point": axis["near_point"],
        "far_point": axis["far_point"],
        "reference_width": int(entry["reference_width"]),
        "reference_height": int(entry["reference_height"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layer1_run",
        description="Layer 1: broad traffic detection, tracking, counting & run summary",
    )
    parser.add_argument("--config", type=Path, default=Path("config/layer1_config.yaml"))
    parser.add_argument("--input", "--video", dest="input", type=Path, default=None, help="Video to process (overrides video.path in the config file)")
    parser.add_argument("--output", "--run-name", dest="output", type=str, required=True, help="Run name; outputs go to <results_dir>/<output>/")
    parser.add_argument("--start-datetime", default=None, help="'%%Y-%%m-%%d %%H:%%M:%%S' (overrides video.start_datetime)")
    parser.add_argument("--camera-key", default=None, help="Look up this camera's ROI + movement_axis from config/roi/roi_by_camera.yaml instead of the config file's static fallback polygon")
    parser.add_argument("--roi-config", type=Path, default=None, help="Override roi.roi_by_camera_path (default: value in the config file)")
    parser.add_argument("--confidence", type=float, default=None, help="Override confidence.default")
    parser.add_argument("--time-bucket-minutes", type=int, default=None, help="Override aggregation.time_bucket_minutes")
    parser.add_argument("--save-crops", action="store_true", default=None, help="Force-enable Bus track crop saving (overrides crops.enabled)")
    parser.add_argument("--debug", action="store_true", default=None, help="Write an annotated debug video (tracked.mp4) with track id/class/confidence/direction overlays")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_layer1_config(args.config)
    except Layer1ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to load config from {args.config}: {e}", file=sys.stderr)
        return 2

    if args.input is not None or args.start_datetime is not None:
        video_path = args.input.resolve() if args.input is not None else config.resolve(config.video.path)
        if not video_path.exists():
            print(f"Error: video not found: {video_path}", file=sys.stderr)
            return 1

        ffprobe_bin = config.resolve(config.tools.ffprobe)
        try:
            probed = probe_video(video_path, ffprobe_bin)
        except Exception as e:
            print(f"Error: could not probe video {video_path}: {e}", file=sys.stderr)
            return 1
        fps = probed.fps if probed.fps and probed.fps > 0 else config.video.fps

        video_cfg = VideoConfig(
            path=str(video_path),
            fps=fps,
            start_datetime=args.start_datetime or config.video.start_datetime,
        )
        config = replace(config, video=video_cfg)

    movement_axis_entry: Optional[dict] = None
    if args.camera_key:
        roi_config_path = args.roi_config or config.resolve(config.roi.roi_by_camera_path)
        try:
            roi_cfg = load_camera_roi(roi_config_path, args.camera_key, config.roi.roi_by_camera_path)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        movement_axis_entry = load_camera_movement_axis(roi_config_path, args.camera_key)
        config = replace(config, roi=roi_cfg)

    if args.confidence is not None:
        config = replace(config, confidence=ConfidenceConfig(default=args.confidence, per_bucket=config.confidence.per_bucket))

    if args.time_bucket_minutes is not None:
        config = replace(config, aggregation=AggregationConfig(time_bucket_minutes=args.time_bucket_minutes))

    try:
        pipeline = Layer1Pipeline(
            config,
            run_name=args.output,
            movement_axis_entry=movement_axis_entry,
            debug=args.debug,
            save_crops=args.save_crops,
        )
    except Exception as e:
        print(f"Error: failed to initialize Layer 1 pipeline: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        pipeline.run()
    except Exception:
        pipeline.logger.error("Run failed with an unhandled exception:\n%s", traceback.format_exc())
        print("Run failed -- see logs for details.", file=sys.stderr)
        return 1

    print(f"\nResults written to: {pipeline.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

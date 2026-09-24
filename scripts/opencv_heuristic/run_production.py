#!/usr/bin/env python
"""Production entrypoint for the multi-class vehicle-counting pipeline:
YAML-config driven, processes either a whole folder of videos (one camera's
worth) or a single video (handy for a quick test run), and writes clean
per-camera, per-15-minute-window unique counts for every vehicle group
(two_wheeler, auto_rickshaw, four_wheeler, bus, truck -- config.py:
VEHICLE_GROUPS), plus a color-classification second layer for buses only.

Usage:
    env/bin/python scripts/opencv_heuristic/run_production.py \
        --config scripts/opencv_heuristic/production_config.example.yaml

See production_config.example.yaml for all options and comments.

Pipeline: YOLO + BoT-SORT detects and tracks every vehicle-group class in
one pass (best with a UVH-26 fine-tune, see uvh26_train.py -- a plain COCO
model only has a generic 'bus' class and can't tell two/three/four-wheelers
apart); tiny/spurious detections are dropped per-group before tracking
(config.py: GROUP_MIN_AREA_FRAC); each bus track is color-classified once
from its single best frame (best_frame.py + classify.py); results are
bucketed into real wall-clock 15-minute windows per camera
(video_discovery.py infers each video's real start time).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import PipelineConfig  # noqa: E402
from detector import load_detector  # noqa: E402
from mode_production import build_class_id_to_group, process_video  # noqa: E402
from run_config import RunConfig  # noqa: E402
from video_discovery import discover_videos, derive_camera_name, parse_video_start_time  # noqa: E402
from video_io import resolve_video  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Path to a YAML run config.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_cfg = RunConfig.from_yaml(args.config)

    cfg = PipelineConfig()
    cfg.DETECTOR_MODEL = run_cfg.detector_model
    cfg.TRACKER_CONFIG = run_cfg.tracker_config
    if run_cfg.detection_confidence is not None:
        cfg.DETECTION_CONFIDENCE = run_cfg.detection_confidence
    if run_cfg.interval_minutes is not None:
        cfg.INTERVAL_MINUTES = run_cfg.interval_minutes
    if run_cfg.imgsz is not None:
        cfg.IMGSZ = run_cfg.imgsz
    if run_cfg.save_best_frame_crops is not None:
        cfg.SAVE_BEST_FRAME_CROPS = run_cfg.save_best_frame_crops

    device = run_cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda:0":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: No GPU in use -- production runs over long/many videos will be slow.")

    output_dir = run_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_cfg.mode == "single_video":
        videos = [run_cfg.input_video]
        print(f"Mode: single_video -- {run_cfg.input_video}")
    else:
        videos = discover_videos(run_cfg.input_folder, run_cfg.video_globs)
        print(f"Mode: folder -- {run_cfg.input_folder} ({len(videos)} video(s) found)")
        if not videos:
            raise SystemExit(f"No videos matching {run_cfg.video_globs} found under {run_cfg.input_folder}")

    detector = load_detector(cfg.DETECTOR_MODEL)
    class_id_to_group = build_class_id_to_group(detector, cfg.VEHICLE_GROUPS)
    if not class_id_to_group:
        raise SystemExit("None of config.py's VEHICLE_GROUPS class names were found in this detector's classes -- "
                          "wrong model for multi-class counting? (a plain COCO model only has 'bus', 'car', "
                          "'truck', 'motorcycle', not the UVH-26 taxonomy).")
    groups_present = sorted(set(class_id_to_group.values()))
    print(f"Vehicle groups tracked: {groups_present} (color layer on: '{cfg.COLOR_CLASSIFIED_GROUP}')")

    all_track_rows = []
    all_window_rows = []
    per_video_stats = []
    run_start = time.time()

    for video_path in videos:
        if not video_path.exists():
            print(f"[WARN] video not found, skipping: {video_path}")
            continue

        camera_name = derive_camera_name(video_path, run_cfg.camera_name)
        start_dt, source = parse_video_start_time(video_path, run_cfg.video_start_times.get(video_path.name))
        if source == "mtime":
            print(f"[WARN] {video_path.name}: could not parse a start time from the filename -- "
                  f"falling back to file mtime ({start_dt.isoformat()}). Window bucketing may be "
                  f"wrong; add an entry under video_start_times in the config to fix it.")
        else:
            print(f"{video_path.name}: start time {start_dt.isoformat()} (source: {source})")

        try:
            resolved_path = resolve_video(video_path, output_dir)
        except Exception as e:
            print(f"[ERROR] {video_path.name}: could not resolve/convert video, skipping: {e}")
            continue

        try:
            result = process_video(detector, class_id_to_group, resolved_path, camera_name,
                                    start_dt, output_dir, cfg, device,
                                    start_offset_seconds=run_cfg.start_offset_seconds,
                                    duration_seconds=run_cfg.duration_seconds)
        except Exception as e:
            print(f"[ERROR] {video_path.name}: processing failed, skipping: {e}")
            continue

        all_track_rows.extend(result["track_rows"])
        all_window_rows.extend(result["window_rows"])
        per_video_stats.append({
            "camera": camera_name, "video": video_path.name,
            "frames": result["frame_count"], "unique_tracks": result["unique_tracks"],
            "rejected_tiny_boxes": result["rejected_tiny_boxes"],
            "processing_seconds": round(result["processing_seconds"], 1),
        })

    total_elapsed = time.time() - run_start

    track_details_path = output_dir / "track_details.csv"
    window_counts_path = output_dir / "window_counts.csv"
    summary_path = output_dir / "run_summary.txt"

    track_columns = ["camera", "video", "track_id", "vehicle_group", "first_seen", "last_seen",
                      "best_frame_wallclock", "best_frame_detection_confidence", "color", "color_confidence"]
    if all_track_rows:
        pd.DataFrame(all_track_rows).sort_values(["camera", "video", "track_id"]).to_csv(track_details_path, index=False)
    else:
        pd.DataFrame(columns=track_columns).to_csv(track_details_path, index=False)
    print(f"\nSaved: {track_details_path} ({len(all_track_rows)} tracks)")

    count_cols = groups_present + ["total_vehicles"] + [f"bus_{c}" for c in cfg.COLOR_CLASSES]
    if all_window_rows:
        window_df = pd.DataFrame(all_window_rows)
        agg_cols = {c: "sum" for c in count_cols if c in window_df.columns}
        # sum across videos that land in the same camera+window (e.g. a folder of
        # back-to-back hourly files); track uniqueness only holds within one video
        # (the tracker's state doesn't persist across files), so this is a sum of
        # already-deduplicated per-video counts, not a re-dedup across files.
        merged = (window_df.groupby(["camera", "window_start", "window_end"], as_index=False)
                  .agg(agg_cols))
        merged = merged.sort_values(["camera", "window_start"])
        merged.to_csv(window_counts_path, index=False)
        window_df.sort_values(["camera", "video", "window_start"]).to_csv(
            output_dir / "window_counts_by_video.csv", index=False)
    else:
        pd.DataFrame(columns=["camera", "window_start", "window_end"] + count_cols).to_csv(
            window_counts_path, index=False)
    print(f"Saved: {window_counts_path}")

    totals_by_group = {g: sum(r.get(g, 0) for r in all_window_rows) for g in groups_present}
    totals_by_bus_color = {c: sum(r.get(f"bus_{c}", 0) for r in all_window_rows) for c in cfg.COLOR_CLASSES}
    total_tracks = len(all_track_rows)
    cameras = sorted({r["camera"] for r in per_video_stats}) if per_video_stats else []

    summary_lines = [
        "MVSA Multi-Class Vehicle Counting Run", "=" * 38, "",
        f"Config: {args.config}",
        f"Mode: {run_cfg.mode}",
        f"Detector model: {cfg.DETECTOR_MODEL}", f"Tracker: BoT-SORT ({cfg.TRACKER_CONFIG})",
        f"Device: {device}", f"Detection confidence: {cfg.DETECTION_CONFIDENCE}",
        f"Inference imgsz: {cfg.IMGSZ if cfg.IMGSZ else '640 (ultralytics default)'}",
        f"Vehicle groups: {groups_present}",
        f"15-minute interval: {cfg.INTERVAL_MINUTES} min", "",
        f"Cameras: {', '.join(cameras) if cameras else '(none processed)'}",
        f"Videos processed: {len(per_video_stats)} / {len(videos)}", "",
        f"Total unique vehicle tracks: {total_tracks}",
    ] + [f"  {g}: {totals_by_group[g]}" for g in groups_present] + [
        "", f"Bus color breakdown (of {totals_by_group.get(cfg.COLOR_CLASSIFIED_GROUP, 0)} buses):",
    ] + [f"  {c}: {totals_by_bus_color[c]}" for c in cfg.COLOR_CLASSES] + [
        "", "Per-video breakdown:",
    ] + [
        f"  [{s['camera']}] {s['video']}: {s['unique_tracks']} tracks, "
        f"{s['rejected_tiny_boxes']} tiny-box rejections, {s['frames']} frames, "
        f"{s['processing_seconds']}s"
        for s in per_video_stats
    ] + [
        "", f"Total processing time: {timedelta(seconds=int(total_elapsed))}", "",
        f"Output directory: {output_dir}",
        "  track_details.csv         -- one row per track: camera, video, vehicle_group, times, "
        "best-frame confidence, color+confidence (buses only)",
        "  window_counts.csv         -- unique counts per group per camera per 15-min window, plus bus_<color>",
        "  window_counts_by_video.csv -- same, broken out per source video",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {summary_path}")

    print("\n" + "\n".join(summary_lines[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

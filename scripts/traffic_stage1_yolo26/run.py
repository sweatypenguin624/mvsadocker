#!/usr/bin/env python
"""Stage 1 CLI: unique-vehicle detection, tracking and 15-minute counting.

Example:
  env/bin/python scripts/traffic_stage1/run.py \
      --video videos/actual_test/08.00.00-09.00.00.mp4 \
      --start-time 2026-07-20T08:00:00 \
      --camera-key cam_1_eb \
      --output results/stage1_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_config import REPO_ROOT, load_config  # noqa: E402
from stage1_pipeline import run_stage1  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "stage1_config.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 1: count unique vehicles per 15-minute window.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--camera-key", default=None, help="Per-camera override block in the config.")
    p.add_argument(
        "--start-time",
        default=None,
        help="Wall-clock time of frame 0 (ISO 8601). Required for correct 15-minute "
             "window labels; inferred from the filename when it looks like HH.MM.SS-HH.MM.SS.",
    )
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames (smoke tests).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def infer_start_time(video: Path) -> datetime:
    """Best-effort start time from Dahua-style names like
    '08.00.00-09.00.00.mp4' under a 'YYYY-MM-DD' parent directory."""
    stem = video.stem.split("-")[0].split("[")[0]
    date = None
    for parent in video.parents:
        try:
            date = datetime.strptime(parent.name, "%Y-%m-%d").date()
            break
        except ValueError:
            continue
    try:
        clock = datetime.strptime(stem, "%H.%M.%S").time()
    except ValueError:
        raise SystemExit(
            f"Cannot infer --start-time from '{video.name}'. Pass --start-time explicitly "
            "(ISO 8601), otherwise the 15-minute window labels will be wrong."
        )
    if date is None:
        raise SystemExit(
            f"Found a clock time in '{video.name}' but no YYYY-MM-DD parent directory. "
            "Pass --start-time explicitly."
        )
    return datetime.combine(date, clock)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    start_time = (
        datetime.fromisoformat(args.start_time) if args.start_time else infer_start_time(args.video)
    )
    config = load_config(args.config if args.config.exists() else None, args.camera_key)
    if not args.config.exists():
        logging.warning("Config %s not found -- using built-in defaults.", args.config)

    summary = run_stage1(
        video_path=args.video,
        output_dir=args.output,
        config=config,
        start_time=start_time,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 64)
    print(f"STAGE 1: {summary['counting']['counted_vehicles']} unique vehicles")
    print("=" * 64)
    if not summary["counting_line"]["calibrated"]:
        print("!! UNCALIBRATED counting line -- see stage1_summary.json")
    t = summary["tracking"]
    print(f"raw tracks {t['raw_tracks']} -> {t['vehicles']} vehicles "
          f"({t['spatial_merges']} spatial + {t['lockstep_merges']} lockstep + "
          f"{t['temporal_merges']} temporal merges)")
    d = summary["detection"]
    print(f"boxes fused away: {d['fused_away']}, dropped by size filter: "
          f"{d['dropped_by_size_filter']}")
    print(f"\n{'interval':<20}{'vehicles':>10}")
    for row in summary["intervals"]:
        print(f"{row['interval_start']:<20}{row['total_vehicles']:>10}")
    print(f"\nCrops for Stage 2: {args.output}/crops/vehicle_*/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

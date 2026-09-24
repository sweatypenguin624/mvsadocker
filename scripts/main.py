#!/usr/bin/env python3
"""MVSA CLI entry point.

Usage:
    python scripts/main.py detect             --config config/config.yaml --run-name my_run
    python scripts/main.py full                --config config/config.yaml --run-name my_run
    python scripts/main.py demographics         --config config/config.yaml --run-name my_run
    python scripts/main.py demographics-test    --config config/config.yaml --run-name my_run

Run from the project root (~/mvsa) so relative paths in config.yaml resolve
correctly, e.g.:

    cd ~/mvsa
    source env/bin/activate
    python scripts/main.py demographics-test --run-name actual_08-09
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from config_loader import ConfigError, load_config
from pipeline import (
    MODE_DEMOGRAPHICS,
    MODE_DEMOGRAPHICS_TEST,
    MODE_DETECT,
    MODE_FULL,
    Pipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvsa",
        description="MVSA pedestrian detection + demographic estimation pipeline",
    )
    parser.add_argument(
        "mode",
        choices=[MODE_DETECT, MODE_FULL, MODE_DEMOGRAPHICS, MODE_DEMOGRAPHICS_TEST],
        help=(
            "detect: YOLO+ByteTrack+ROI pedestrian counting only. "
            "full: detect + inline demographics in one pass. "
            "demographics: second-pass demographics from a prior detect/full run "
            "(no re-running YOLO). "
            "demographics-test: full pipeline capped at a small number of tracks, "
            "for validating whether this camera's footage supports age/gender "
            "estimation before scaling up."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of this run; outputs go to <results_dir>/<run-name>/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from results/<run-name>/checkpoint.json if present "
        "(detect/full/demographics-test only; see README for resume limitations)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to load config from {args.config}: {e}", file=sys.stderr)
        return 2

    pipeline = Pipeline(config, run_name=args.run_name, resume=args.resume)

    try:
        if args.mode == MODE_DETECT:
            summary = pipeline.run_detect()
        elif args.mode == MODE_FULL:
            summary = pipeline.run_full()
        elif args.mode == MODE_DEMOGRAPHICS:
            summary = pipeline.run_demographics()
        elif args.mode == MODE_DEMOGRAPHICS_TEST:
            summary = pipeline.run_demographics_test()
        else:
            parser.error(f"unknown mode: {args.mode}")
            return 2
    except Exception:
        pipeline.logger.error("Run failed with an unhandled exception:\n%s", traceback.format_exc())
        return 1

    pipeline.logger.info("Run complete. Summary: %s", summary)
    print(f"Done. Results written to: {pipeline.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

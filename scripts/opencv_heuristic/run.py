#!/usr/bin/env python
"""CLI for the bus color classification pipeline (local re-implementation of
the CPM Hubli-Dharwad Colab notebook).

Three modes, same as the notebook's RUN_MODE:

  collect_crops     Sample bus crops per track for manual labeling.
  train_classifier  Fine-tune a YOLO classification head on labeled crops.
  full_pipeline     Track buses, classify color with the OpenCV HSV
                     heuristic, and aggregate into 15-minute interval counts.

Examples:
  env/bin/python scripts/opencv_heuristic/run.py collect_crops \
      --video videos/actual_test/08.00.00-09.00.00.mp4 \
      --output results/bus_color/unlabeled_crops

  env/bin/python scripts/opencv_heuristic/run.py train_classifier \
      --color-dataset-dir results/bus_color/bus_color_dataset

  env/bin/python scripts/opencv_heuristic/run.py full_pipeline \
      --video videos/actual_test/08.00.00-09.00.00.mp4 \
      --output results/bus_color/run_2026_09_02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import PipelineConfig  # noqa: E402
from detector import load_detector, resolve_bus_class_id  # noqa: E402
from mode_collect_crops import run_collect_crops  # noqa: E402
from mode_full_pipeline import run_full_pipeline  # noqa: E402
from mode_train_classifier import run_train_classifier  # noqa: E402
from video_io import read_video_meta, resolve_video  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["collect_crops", "train_classifier", "full_pipeline"])
    p.add_argument("--video", type=Path, help="Input video (required for collect_crops/full_pipeline). "
                                                 "A .dav input is converted to .mp4 first via ffmpeg.")
    p.add_argument("--output", type=Path, required=True,
                    help="Output directory (crops dir for collect_crops, run dir for full_pipeline).")
    p.add_argument("--detector-model", default=PipelineConfig.DETECTOR_MODEL)
    p.add_argument("--bus-class-id", type=int, default=None, help="Override auto-detected 'bus' class id.")
    p.add_argument("--device", default=None, help="cuda:0 or cpu; auto-detected if omitted.")
    p.add_argument("--color-dataset-dir", type=Path, default=None,
                    help="For train_classifier: dir containing train/red,blue,yellow,rest.")
    p.add_argument("--save-annotated-video", action="store_true")
    p.add_argument("--save-debug-crops", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = PipelineConfig()
    cfg.SAVE_ANNOTATED_VIDEO = args.save_annotated_video
    cfg.SAVE_DEBUG_CROPS = args.save_debug_crops

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda:0":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: No GPU in use -- this will be slow for full_pipeline/collect_crops.")

    if args.mode == "train_classifier":
        if args.color_dataset_dir is None:
            raise SystemExit("--color-dataset-dir is required for mode=train_classifier")
        run_train_classifier(args.color_dataset_dir, cfg, device)
        return 0

    if args.video is None:
        raise SystemExit(f"--video is required for mode={args.mode}")

    args.output.mkdir(parents=True, exist_ok=True)
    video_path = resolve_video(args.video, args.output)
    read_video_meta(video_path)

    detector = load_detector(args.detector_model)
    bus_class_id = resolve_bus_class_id(detector, args.bus_class_id)

    if args.mode == "collect_crops":
        run_collect_crops(detector, bus_class_id, video_path, args.output, cfg, device)
    else:
        run_full_pipeline(detector, bus_class_id, video_path, args.output, cfg, device, args.detector_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Pick the best frames out of a video and score every sampled frame.

Samples frames from a video, scores each on sharpness, exposure,
contrast, colourfulness and entropy (see frame_quality.py), and selects
the top-K best frames, enforcing a minimum time gap and perceptual
dedupe so the picks aren't clustered around one moment. Writes the
selected frames as images plus a JSON report of every sampled frame's
score.

Example:
  env/bin/python scripts/frame_selection/run.py \
      --video videos/example.mp4 \
      --output results/frame_selection/example \
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from frame_quality import FrameQualityScorer, QualityWeights  # noqa: E402
from frame_selector import (  # noqa: E402
    VideoFrameAnalyzer,
    save_selected_frames,
    select_best_frames,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Select the best frames from a video and score every sampled frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Directory for images + report.")
    p.add_argument("--top-k", type=int, default=5, help="Number of best frames to select.")
    p.add_argument("--max-samples", type=int, default=300, help="Cap on frames analyzed (auto-strides).")
    p.add_argument("--stride", type=int, default=None, help="Score every Nth frame (overrides --max-samples).")
    p.add_argument("--min-score", type=float, default=0.0, help="Discard candidates below this score (0-100).")
    p.add_argument("--min-gap-seconds", type=float, default=1.0, help="Minimum time between selected frames.")
    p.add_argument(
        "--dedupe-hamming-threshold",
        type=int,
        default=6,
        help="Max perceptual-hash distance (0-64) to treat two frames as duplicates.",
    )
    p.add_argument("--resize-width", type=int, default=960, help="Downscale width used for scoring.")
    p.add_argument("--start-time", type=float, default=0.0, help="Seconds into the video to start sampling.")
    p.add_argument("--end-time", type=float, default=None, help="Seconds into the video to stop sampling.")
    p.add_argument("--sharpness-weight", type=float, default=0.35)
    p.add_argument("--exposure-weight", type=float, default=0.25)
    p.add_argument("--contrast-weight", type=float, default=0.15)
    p.add_argument("--colorfulness-weight", type=float, default=0.10)
    p.add_argument("--entropy-weight", type=float, default=0.15)
    p.add_argument("--no-save-images", action="store_true", help="Only write the JSON report.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    weights = QualityWeights(
        sharpness=args.sharpness_weight,
        exposure=args.exposure_weight,
        contrast=args.contrast_weight,
        colorfulness=args.colorfulness_weight,
        entropy=args.entropy_weight,
    )
    scorer = FrameQualityScorer(weights=weights)
    analyzer = VideoFrameAnalyzer(scorer=scorer, resize_width=args.resize_width)

    logging.info("Analyzing %s ...", args.video)
    results = analyzer.analyze(
        str(args.video),
        stride=args.stride,
        max_samples=args.max_samples,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    logging.info("Scored %d sampled frames.", len(results))

    selected = select_best_frames(
        results,
        top_k=args.top_k,
        min_score=args.min_score,
        min_gap_seconds=args.min_gap_seconds,
        dedupe_hamming_threshold=args.dedupe_hamming_threshold,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    written_images = []
    if not args.no_save_images and selected:
        written_images = save_selected_frames(str(args.video), selected, str(args.output))

    report = {
        "video": str(args.video),
        "frames_analyzed": len(results),
        "selected": [r.as_dict() for r in sorted(selected, key=lambda r: r.score, reverse=True)],
        "all_scores": [r.as_dict() for r in results],
    }
    report_path = args.output / "frame_scores.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print(f"FRAME SELECTION: {len(selected)}/{len(results)} sampled frames selected")
    print("=" * 64)
    print(f"{'rank':<6}{'frame':>10}{'time(s)':>10}{'score':>10}")
    for rank, r in enumerate(sorted(selected, key=lambda r: r.score, reverse=True), start=1):
        print(f"{rank:<6}{r.index:>10}{r.timestamp:>10.2f}{r.score:>10.2f}")
    if written_images:
        print(f"\nImages: {args.output}/")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""YOLO26 experiment: single-pass detect+fuse+track with a rendered annotated
video, for visual QC of a run.

Not part of Stage 1 proper (that stays two-pass and class-agnostic, with no
video output). This is a lightweight companion that reuses the same
detector/fusion/tracker modules to draw what the model saw, plus a
real-time top-right HUD of counted vehicles by class. Counting here is a
simple single-pass line crossing per track id (no identity-merge, no
dwell/interval logic) -- for the calibrated two-pass vehicle count, use
run.py and read stage1_summary.json.

Class buckets: YOLO26's weights are COCO-pretrained, which has no
three-wheeler/auto-rickshaw class, so that bucket is always 0 here (shown
with a note) -- it is not something this detector can distinguish. Bicycle
and motorcycle are folded into "2-Wheeler"; car, bus and truck are kept
separate since they're visually and behaviourally distinct.

Example:
  env/bin/python scripts/traffic_stage1_yolo26/annotate_run.py \
      --video videos/actual_test/20260720_090553_tp00075.mp4 \
      --config scripts/traffic_stage1_yolo26/stage1_config_yolo26.yaml \
      --camera-key tp00075 \
      --output results/stage1_yolo26_9to915/annotated.mp4 \
      --max-frames 18000
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_config import load_config  # noqa: E402
from stage1_detect import VehicleDetector  # noqa: E402
from stage1_fusion import fuse_boxes  # noqa: E402
from stage1_geometry import ground_point, iou, side_of_line  # noqa: E402
from stage1_track import VehicleTracker  # noqa: E402

logger = logging.getLogger("mvsa.traffic_stage1_yolo26.annotate")

# COCO class name -> HUD bucket. Anything not listed here is dropped
# already by the detector's include_classes allowlist.
CLASS_BUCKETS = {
    "bicycle": "2-Wheeler",
    "motorcycle": "2-Wheeler",
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck",
}
# Display order for the HUD, plus the always-zero 3-wheeler note.
BUCKET_ORDER = ["2-Wheeler", "3-Wheeler", "Car", "Bus", "Truck"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render an annotated video for a YOLO26 Stage 1 run.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Path to write annotated .mp4")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--camera-key", default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--line-y-frac", type=float, default=0.6, help="Fallback horizontal line, as a fraction of frame height.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def draw_hud(frame, w: int, total: int, counts: dict) -> None:
    lines = [f"Total: {total}"]
    for bucket in BUCKET_ORDER:
        if bucket == "3-Wheeler":
            lines.append("3-Wheeler: N/A (not in COCO classes)")
        else:
            lines.append(f"{bucket}: {counts.get(bucket, 0)}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, line_h, pad = 0.65, 2, 26, 12
    text_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines)
    box_w = text_w + 2 * pad
    box_h = len(lines) * line_h + 2 * pad
    x1, y1 = w - box_w - 15, 15
    x2, y2 = w - 15, 15 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    ty = y1 + pad + 18
    for i, text in enumerate(lines):
        color = (255, 255, 255) if i > 0 else (0, 255, 255)
        cv2.putText(frame, text, (x1 + pad, ty + i * line_h), font, scale, color, thick)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    config = load_config(args.config, args.camera_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    line_pts = config.counting.line
    if not line_pts:
        y = int(h * args.line_y_frac)
        line_pts = [[0.0, float(y)], [float(w), float(y)]]
        logger.warning("No calibrated line for this camera -- using fallback at y=%d", y)
    else:
        rw, rh = config.counting.reference_width, config.counting.reference_height
        if rw and rh and (rw, rh) != (w, h):
            sx, sy = w / rw, h / rh
            line_pts = [[p[0] * sx, p[1] * sy] for p in line_pts]
    p1, p2 = line_pts[0], line_pts[1]

    detector = VehicleDetector(config)
    tracker = VehicleTracker(config.tracker, frame_rate=int(round(fps)))
    class_names = detector.names

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    last_side = {}
    crossed_ids = set()
    class_votes: dict[int, Counter] = {}
    bucket_counts: Counter = Counter()
    total = 0
    frame_idx = 0
    processed = 0

    while True:
        if args.max_frames and processed >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        boxes, confs, cls_ids = detector.detect_with_classes(frame)
        fused_boxes, fused_confs, groups = fuse_boxes(boxes, confs, config.fusion)
        # Each fused box's class = the class of its highest-confidence member.
        fused_cls = [
            cls_ids[max(members, key=lambda m: confs[m])] for members in groups
        ]
        tracks = tracker.update(fused_boxes, fused_confs, frame)

        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)

        for track_id, bbox, conf in tracks:
            # Match this track's (possibly Kalman-adjusted) box back to the
            # fused detection it came from, by best IoU, to read off a class.
            best_j, best_iou = -1, 0.0
            for j, fb in enumerate(fused_boxes):
                score = iou(bbox, fb)
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_j >= 0 and best_iou > 0.1:
                cname = str(class_names.get(int(fused_cls[best_j]), "?")).lower()
                class_votes.setdefault(track_id, Counter())[cname] += 1

            gp = ground_point(bbox)
            side = side_of_line(gp, p1, p2)
            prev = last_side.get(track_id)
            just_crossed = prev is not None and (prev > 0) != (side > 0) and track_id not in crossed_ids
            last_side[track_id] = side

            if just_crossed:
                crossed_ids.add(track_id)
                total += 1
                votes = class_votes.get(track_id)
                cname = votes.most_common(1)[0][0] if votes else None
                bucket = CLASS_BUCKETS.get(cname)
                if bucket:
                    bucket_counts[bucket] += 1

            color = (0, 200, 0) if track_id in crossed_ids else (0, 165, 255)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            votes = class_votes.get(track_id)
            label_cls = votes.most_common(1)[0][0] if votes else "?"
            cv2.putText(
                frame, f"ID:{track_id} {label_cls} {conf:.2f}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        draw_hud(frame, w, total, bucket_counts)
        writer.write(frame)
        frame_idx += 1
        processed += 1
        if processed % 500 == 0:
            logger.info("%d frames rendered, %d crossed so far", processed, total)

    cap.release()
    writer.release()
    logger.info(
        "Wrote %d frames to %s (%d crossings, by class: %s)",
        processed, args.output, total, dict(bucket_counts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

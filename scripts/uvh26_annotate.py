#!/usr/bin/env python3
"""Run the UVH-26 Indian-traffic vehicle detector on sampled video frames
and write YOLO-format label files for human correction ahead of
fine-tuning.

Output layout (flat, video-stem-prefixed filenames so multiple videos can
share one dataset dir, matching dataset/stest/'s convention):

    <output-dir>/
      images/<stem>_f<frame_idx>.jpg    # raw frame, to correct labels against
      labels/<stem>_f<frame_idx>.txt    # YOLO format: cls xc yc w h (normalized)
      preview/<stem>_f<frame_idx>.jpg   # frame with predicted boxes drawn, for quick QC
      classes.txt                        # class id -> name, model order
      data.yaml                          # draft ultralytics dataset config

Usage:
    python scripts/uvh26_annotate.py \
        --video videos/sample.mp4 videos/actual_test/08.00.00-09.00.00.mp4 \
        --interval-seconds 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvh26_annotate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "models" / "UVH-26" / "weights" / "YOLOv11-X" / "UVH-26-MV-YOLOv11-X.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "uvh26_review"

BOX_COLOR = (60, 200, 60)


def sample_frame_indices(frame_count: int, fps: float, interval_seconds: float) -> List[int]:
    if fps <= 0:
        raise ValueError("video reports fps <= 0, cannot compute frame interval")
    step = max(1, round(fps * interval_seconds))
    return list(range(0, frame_count, step))


def annotate_video(
    video_path: Path,
    model,
    output_dir: Path,
    interval_seconds: float,
    conf: float,
    device: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, fps, interval_seconds)

    stem = video_path.stem
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"
    for d in (images_dir, labels_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            logger.warning("failed to read frame %d of %s, skipping", idx, video_path.name)
            continue

        h, w = frame.shape[:2]
        name = f"{stem}_f{idx:07d}"

        results = model(frame, conf=conf, device=device, verbose=False)
        result = results[0]

        lines = []
        preview = frame.copy()
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
                label = f"{model.names[cls_id]} {float(box.conf[0]):.2f}"
                cv2.putText(
                    preview,
                    label,
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    BOX_COLOR,
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(images_dir / f"{name}.jpg"), frame)
        cv2.imwrite(str(preview_dir / f"{name}.jpg"), preview)
        (labels_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    cap.release()
    logger.info(
        "%s: wrote %d annotated frames (of %d total, every %.1fs)",
        video_path.name,
        written,
        frame_count,
        interval_seconds,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", nargs="+", required=True, type=Path, help="one or more input video files")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = model.names
    (args.output_dir / "classes.txt").write_text(
        "\n".join(names[i] for i in sorted(names)) + "\n"
    )

    data_yaml = args.output_dir / "data.yaml"
    if not data_yaml.exists():
        names_block = "\n".join(f"  {i}: {names[i]}" for i in sorted(names))
        data_yaml.write_text(
            "# Draft dataset config for fine-tuning UVH-26 after manual label review.\n"
            "# labels/ holds model predictions -- correct them by hand, then split\n"
            "# into train/valid subdirs (see dataset/stest/ for the expected layout)\n"
            "# before training.\n"
            f"path: {args.output_dir.resolve()}\n\n"
            "train: images\n"
            "val: images\n\n"
            f"nc: {len(names)}\n\n"
            f"names:\n{names_block}\n"
        )

    total = 0
    for video_path in args.video:
        total += annotate_video(
            video_path, model, args.output_dir, args.interval_seconds, args.conf, args.device
        )

    logger.info(
        "done: %d frames written across %d video(s) -> %s",
        total,
        len(args.video),
        args.output_dir,
    )


if __name__ == "__main__":
    main()

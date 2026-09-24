"""Auto-sort bus crops into a color dataset using the OpenCV HSV heuristic.

Unlike sort_crops.py (manual, browser-based labeling), this runs
classify_bus_crop() from classify.py on every crop and files it straight
into <dataset-dir>/train/<color>/ -- no human in the loop. Useful for
bootstrapping a dataset fast; expect some mislabeled crops since it's the
same heuristic full_pipeline uses, not ground truth.

Usage:
    env/bin/python scripts/opencv_heuristic/auto_classify_crops.py \\
        --crops-dir results/bus_color/unlabeled_crop \\
        --dataset-dir results/bus_color/bus_color_dataset

By default crops are copied (source dir left intact); pass --move to move
them instead, same as sort_crops.py does.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cv2  # noqa: E402

from classify import classify_bus_crop  # noqa: E402
from config import PipelineConfig  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops-dir", type=Path, required=True,
                    help="Dir of unlabeled bus crop images, e.g. results/bus_color/unlabeled_crop")
    p.add_argument("--dataset-dir", type=Path, required=True,
                    help="Where to write train/red,blue,yellow,rest.")
    p.add_argument("--move", action="store_true", help="Move crops instead of copying them.")
    p.add_argument("--min-confidence", type=float, default=0.0,
                    help="Crops below this confidence are filed under 'rest' regardless of the "
                         "heuristic's winning color (default: 0.0, i.e. no override).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.crops_dir.is_dir():
        raise SystemExit(f"--crops-dir not found: {args.crops_dir}")

    cfg = PipelineConfig()
    train_dir = args.dataset_dir / "train"
    for color in cfg.COLOR_CLASSES:
        (train_dir / color).mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.crops_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No image files found in {args.crops_dir}")

    counts = {c: 0 for c in cfg.COLOR_CLASSES}
    failed = 0
    transfer = shutil.move if args.move else shutil.copy2

    for i, path in enumerate(paths, 1):
        crop = cv2.imread(str(path))
        color, confidence = classify_bus_crop(crop, cfg)
        if color is None:
            failed += 1
            continue
        if confidence is not None and confidence < args.min_confidence:
            color = cfg.FALLBACK_COLOR

        dst = train_dir / color / path.name
        transfer(str(path), str(dst))
        counts[color] += 1

        if i % 100 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {counts}")

    print(f"\nDone. {sum(counts.values())} crops classified, {failed} failed to load.")
    for color, n in counts.items():
        print(f"  {color}: {n}")
    print(f"Dataset dir: {train_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

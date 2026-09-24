"""CLI entrypoint for Step 1 of the vehicle subtype classifier.

Usage:
    python scripts/vehicle_subtype_classifier/run.py \
        --frames-dir /path/to/extracted_frames \
        --output-dir results/subtype_step1_run1 \
        [--config config/vehicle_subtype_config.yaml]

See INPUT_REQUIREMENTS.md in this directory for what --frames-dir must
contain.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bus_subclass_classifier import BusSubclassClassifier  # noqa: E402
from subtype_classifier import SubtypeClassifier  # noqa: E402
from subtype_config import load_config  # noqa: E402

REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "vehicle_subtype_config.yaml"
DEFAULT_BUS_SUBCLASS_CONFIG = REPO_ROOT / "config" / "bus_subclass_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
BUS_NATIVE_CLASSES = {"Bus", "Mini-bus"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vehicle_subtype_classifier")


def find_frames(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bus-subclass-config", type=Path, default=DEFAULT_BUS_SUBCLASS_CONFIG)
    parser.add_argument(
        "--no-bus-subclass",
        action="store_true",
        help="Skip bus subclass classification even for Bus/Mini-bus frames.",
    )
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        raise SystemExit(f"--frames-dir does not exist or is not a directory: {args.frames_dir}")

    frame_paths = find_frames(args.frames_dir)
    if not frame_paths:
        raise SystemExit(
            f"No images found under {args.frames_dir} "
            f"(looked for {sorted(IMAGE_EXTENSIONS)}, recursively)"
        )
    logger.info("Found %d frames under %s", len(frame_paths), args.frames_dir)

    config = load_config(args.config)
    classifier = SubtypeClassifier(config)
    bus_subclassifier = (
        None if args.no_bus_subclass else BusSubclassClassifier(args.bus_subclass_config)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / config.output.results_filename
    summary_path = args.output_dir / config.output.summary_filename

    class_counts: Counter = Counter()
    bus_subclass_counts: Counter = Counter()
    unreadable: list[str] = []
    start = time.time()

    with open(results_path, "w") as out_f:
        for i, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning("Could not read image, skipping: %s", frame_path)
                unreadable.append(str(frame_path))
                continue

            rel_path = str(frame_path.relative_to(args.frames_dir))
            result = classifier.classify_frame(frame, rel_path)
            class_counts[result.label] += 1

            record = {
                "frame": result.frame_path,
                "class": result.label,
                "confidence": round(result.confidence, 4),
                "bbox": result.bbox,
                "num_detections": result.num_detections,
            }

            if bus_subclassifier is not None and result.label in BUS_NATIVE_CLASSES:
                if result.bbox is not None:
                    x1, y1, x2, y2 = (int(round(v)) for v in result.bbox)
                    crop = frame[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
                else:
                    crop = frame  # no box (shouldn't happen when label != Unclassified)
                bus_result = bus_subclassifier.classify(crop, result.label)
                bus_subclass_counts[bus_result.subclass] += 1
                record["bus_subclass"] = bus_result.subclass
                record["bus_subclass_confidence"] = round(bus_result.confidence, 4)
                record["bus_subclass_reason"] = bus_result.reason

            out_f.write(json.dumps(record) + "\n")

            if i % 200 == 0:
                logger.info("Processed %d/%d frames", i, len(frame_paths))

    elapsed = time.time() - start
    summary = {
        "frames_dir": str(args.frames_dir),
        "total_frames_found": len(frame_paths),
        "frames_classified": len(frame_paths) - len(unreadable),
        "unreadable_frames": unreadable,
        "class_counts": dict(class_counts),
        "bus_subclass_counts": dict(bus_subclass_counts),
        "elapsed_seconds": round(elapsed, 2),
        "model_path": str(config.model.path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done in %.1fs. Results: %s Summary: %s", elapsed, results_path, summary_path)
    logger.info("Class counts: %s", dict(class_counts))
    if bus_subclass_counts:
        logger.info("Bus subclass counts: %s", dict(bus_subclass_counts))


if __name__ == "__main__":
    main()

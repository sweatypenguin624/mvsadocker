"""Fine-tune a YOLO26 detector on UVH-26 so buses stop being confused with
auto-rickshaws.

Why this exists: the COCO-pretrained detector has no 'three-wheeler' class, so
an auto-rickshaw's nearest COCO neighbour is often 'bus' -- and it says so
confidently (0.6-0.9), which no confidence threshold or cross-class
suppression heuristic could fix. UVH-26 has Three-wheeler, Mini-bus and
Tempo-traveller as classes distinct from Bus, so a model trained on it can
simply tell them apart.

Defaults are chosen from the dataset's own statistics (see uvh26_prepare.py
output): images are uniformly 1920x1080; median box is ~113px wide, and Bus
(~231px) / Three-wheeler (~112px) are both comfortably resolved at imgsz=960,
which keeps a 5th-percentile 34px box at a still-detectable ~17px.

Usage:
    env/bin/python scripts/opencv_heuristic/uvh26_train.py \\
        --data data/uvh26/uvh26.yaml --model yolo26s.pt --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/uvh26/uvh26.yaml"))
    p.add_argument("--model", default="yolo26s.pt", help="pretrained checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--patience", type=int, default=15, help="early-stop after N epochs without improvement")
    p.add_argument("--device", default="0")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                    help="cache decoded images after the first epoch so full-res PNGs aren't re-decoded from "
                         "disk every epoch -- this is the single biggest lever on wall-clock time here")
    p.add_argument("--project", type=Path, default=Path("results/uvh26_training"))
    p.add_argument("--name", default="vehiclenet_y26s")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        raise SystemExit(f"--data not found: {args.data} (run uvh26_prepare.py first)")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        cache=False if args.cache == "false" else args.cache,
        # ultralytics nests a relative `project` under its own runs/<task>/ root --
        # resolve to an absolute path so output actually lands where we said.
        project=str(args.project.resolve()),
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        val=True,
        plots=True,
    )

    best = args.project / args.name / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best}")
    print("Point the production pipeline at it with, in the run config:")
    print(f"  detector_model: {best}")
    print(f"  imgsz: {args.imgsz}   # match training resolution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

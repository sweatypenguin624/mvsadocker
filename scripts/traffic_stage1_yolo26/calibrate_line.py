#!/usr/bin/env python
"""Draw / record the counting line for a camera.

Without a calibrated line Stage 1 falls back to a horizontal line at 60%
of frame height, which counts *something* but is not tied to the road, so
every camera used in production should be calibrated once with this tool.

Two modes:
  --extract-frame   save a reference frame to annotate (headless-friendly)
  --set-line        write a line into config/stage1_config.yaml for a camera
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage1_config import REPO_ROOT  # noqa: E402


def extract_frame(video: Path, out: Path, at_frame: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, at_frame)
    ok, frame = cap.read()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {at_frame} of {video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Wrote {out} ({w}x{h}).")
    print("Open it, read off the two endpoint pixel coordinates of your counting")
    print("line across the road, then run:")
    print(f"  calibrate_line.py --set-line --camera-key <key> --line x1,y1,x2,y2 "
          f"--reference-size {w},{h}")


def set_line(config_path: Path, camera_key: str, line, ref_w: int, ref_h: int) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # NOTE: no "or {}" after setdefault -- setdefault already guarantees a
    # dict, and an EMPTY dict is falsy in Python, so "x or {}" silently
    # discards the dict just created/found and mutates a disconnected copy.
    cameras = data.setdefault("cameras", {})
    if cameras is None:
        cameras = data["cameras"] = {}
    block = cameras.setdefault(camera_key, {})
    if block is None:
        block = cameras[camera_key] = {}
    counting = block.setdefault("counting", {})
    if counting is None:
        counting = block["counting"] = {}
    counting["line"] = [[line[0], line[1]], [line[2], line[3]]]
    counting["reference_width"] = ref_w
    counting["reference_height"] = ref_h
    cameras[camera_key] = block
    data["cameras"] = cameras
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Set counting line for '{camera_key}' in {config_path}: {counting['line']} "
          f"(reference {ref_w}x{ref_h})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract-frame", action="store_true")
    p.add_argument("--set-line", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--at-frame", type=int, default=0)
    p.add_argument("--camera-key")
    p.add_argument("--line", help="x1,y1,x2,y2 in reference-size pixels")
    p.add_argument("--reference-size", default="1920,1080", help="W,H the coords were read at")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "stage1_config.yaml")
    args = p.parse_args(argv)

    if args.extract_frame:
        if not args.video:
            raise SystemExit("--extract-frame needs --video")
        out = args.out or (REPO_ROOT / "config" / "roi" / "frames" / f"{args.video.stem}_stage1.jpg")
        extract_frame(args.video, out, args.at_frame)
        return 0

    if args.set_line:
        if not (args.camera_key and args.line):
            raise SystemExit("--set-line needs --camera-key and --line x1,y1,x2,y2")
        coords = [float(v) for v in args.line.split(",")]
        if len(coords) != 4:
            raise SystemExit("--line must be x1,y1,x2,y2")
        ref = [int(v) for v in args.reference_size.split(",")]
        set_line(args.config, args.camera_key, coords, ref[0], ref[1])
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

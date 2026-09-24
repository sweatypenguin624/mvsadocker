#!/usr/bin/env python3
"""Check that the NVDEC reader yields exactly the frames the OpenCV reader does.

Exit code 0 only if frame count and every pixel match. Run this once per
server/ffmpeg build before setting processing.decoder: nvdec.

    python scripts/l40s/compare_decoders.py VIDEO.mp4 [--max-frames 3000] [--ffmpeg ffmpeg]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vehicle-counting" / "pipeline" / "counting"))

from video_utils import VideoStreamReader  # noqa: E402
from nvdec_reader import NvdecVideoReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    with VideoStreamReader(Path(args.video)) as cpu:
        w, h = cpu.width, cpu.height
        with NvdecVideoReader(Path(args.video), w, h, ffmpeg_bin=args.ffmpeg, gpu_id=args.gpu) as gpu:
            print("ffmpeg:", " ".join(gpu.command()))
            a_it, b_it = cpu.frames(), gpu.frames()
            n = mismatched = worst = 0
            first_bad = None
            while not args.max_frames or n < args.max_frames:
                a, b = next(a_it, None), next(b_it, None)
                if a is None or b is None:
                    if (a is None) != (b is None):
                        print(f"FRAME COUNT MISMATCH after {n} frames "
                              f"(opencv ended={a is None}, nvdec ended={b is None})")
                        return 1
                    break
                d = int(np.abs(a[1].astype(np.int16) - b[1].astype(np.int16)).max())
                if d:
                    mismatched += 1
                    worst = max(worst, d)
                    first_bad = first_bad if first_bad is not None else n
                n += 1
                if n % 1000 == 0:
                    print(f"  {n} frames compared, {mismatched} differ")

    print(f"compared {n} frames: {mismatched} differ, max abs pixel diff {worst}"
          + (f", first at frame {first_bad}" if first_bad is not None else ""))
    if mismatched:
        print("FAIL: keep processing.decoder: opencv on this server")
        return 1
    print("PASS: NVDEC output is pixel-identical; processing.decoder: nvdec is safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""NVDEC (GPU hardware) video decoding through an ffmpeg subprocess.

HEVC/H.264 decoding is bit-exact by spec, so NVDEC yields the same YUV planes
as the CPU decoder. The YUV->BGR step is configured to mirror OpenCV's FFmpeg
backend (swscale, yuv420p source, BICUBIC flags, BT.601 matrix, range taken
from the stream). Timestamps pass through untouched so no frame is duplicated
or dropped for VFR (DAV-sourced) footage. Pixel-equality with the OpenCV
reader must still be confirmed per ffmpeg build with
scripts/l40s/compare_decoders.py before enabling it in production.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


class NvdecVideoReader:
    def __init__(self, video_path: Path, width: int, height: int,
                 ffmpeg_bin: str = "ffmpeg", gpu_id: int = 0):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"video file not found: {self.video_path}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid frame size {width}x{height} for {self.video_path}")
        self.width, self.height = width, height
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.gpu_id = gpu_id
        self.proc = None
        self._stderr = None
        self.frames_decoded = 0

    def command(self) -> list[str]:
        return [
            self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-hwaccel", "cuda", "-hwaccel_device", str(self.gpu_id),
            "-i", str(self.video_path),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vsync", "passthrough",
            "-vf", "format=yuv420p,scale=flags=bicubic:in_color_matrix=bt601",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]

    def __enter__(self) -> "NvdecVideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def _read_into(self, buf: np.ndarray) -> bool:
        view = memoryview(buf).cast("B")
        got = 0
        while got < len(view):
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                if got:
                    raise RuntimeError(f"truncated frame from ffmpeg ({got}/{len(view)} bytes)")
                return False
            got += n
        return True

    def frames(self, start_frame: int = 0, keep=None) -> Iterator[Tuple[int, np.ndarray]]:
        # Decode from the start and drop frames before start_frame: ffmpeg -ss is
        # timestamp-based and not frame-exact on VFR streams.
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=self._stderr,
                                     stdin=subprocess.DEVNULL, bufsize=0)
        self.frames_decoded = 0
        shape = (self.height, self.width, 3)
        scratch = np.empty(shape, np.uint8)
        idx = 0
        while True:
            wanted = idx >= start_frame and (keep is None or keep(idx))
            frame = np.empty(shape, np.uint8) if wanted else scratch
            if not self._read_into(frame):
                break
            self.frames_decoded += 1
            if wanted:
                yield idx, frame
            idx += 1
        rc = self.proc.wait()
        if rc != 0 and self.frames_decoded == 0:
            self._stderr.seek(0)
            raise RuntimeError(f"ffmpeg NVDEC decode failed (exit {rc}): "
                               f"{self._stderr.read().decode(errors='replace')[-2000:]}")

    def release(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self._stderr:
            self._stderr.close()
            self._stderr = None

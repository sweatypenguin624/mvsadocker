"""Video I/O: DAV->MP4 remuxing via ffmpeg, ffprobe inspection, and a
streaming frame reader that never loads a full video into memory.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional, Tuple
import threading
import queue
import time

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def remux_dav_to_mp4(dav_path: Path, mp4_path: Path, ffmpeg_bin: Path) -> None:
    """Remux a .dav CCTV file into an .mp4 container without re-encoding.

    Uses ``-c:v copy`` so the original HEVC bitstream is preserved exactly
    (per project requirement -- do not re-encode source video). Audio, if
    present, is dropped since these are silent CCTV feeds and audio codecs
    inside .dav containers are frequently unsupported by MP4.
    """
    dav_path = Path(dav_path)
    mp4_path = Path(mp4_path)
    ffmpeg_bin = Path(ffmpeg_bin)

    if not dav_path.exists():
        raise FileNotFoundError(f"source DAV file not found: {dav_path}")
    if not ffmpeg_bin.exists():
        raise FileNotFoundError(f"ffmpeg binary not found: {ffmpeg_bin}")

    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(dav_path),
        "-c:v",
        "copy",
        "-an",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg remux failed (exit {result.returncode}) for {dav_path}:\n{result.stderr}"
        )


def probe_video(video_path: Path, ffprobe_bin: Path) -> VideoInfo:
    """Inspect a video file with ffprobe and return its key properties."""
    video_path = Path(video_path)
    ffprobe_bin = Path(ffprobe_bin)

    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if not ffprobe_bin.exists():
        raise FileNotFoundError(f"ffprobe binary not found: {ffprobe_bin}")

    cmd = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    stream = streams[0]

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    # Prefer avg_frame_rate (actual decoded rate) over r_frame_rate (container
    # tbr) -- for DAV-sourced CCTV footage with variable frame timestamps,
    # r_frame_rate can report a container time-base (e.g. 100) wildly higher
    # than the true playback rate, which corrupts every frame_idx -> wall-clock
    # timestamp computed downstream (interval binning, OCR start-time fallback).
    def _parse_rate(raw: str) -> float:
        num, _, den = raw.partition("/")
        try:
            return float(num) / float(den) if den and float(den) != 0 else float(num)
        except (ValueError, ZeroDivisionError):
            return 0.0

    fps = _parse_rate(stream.get("avg_frame_rate", "0/1"))
    if fps <= 0:
        fps = _parse_rate(stream.get("r_frame_rate", "0/1"))

    duration = float(stream.get("duration", 0.0) or 0.0)
    nb_frames_raw = stream.get("nb_frames")
    if nb_frames_raw is not None:
        frame_count = int(nb_frames_raw)
    elif duration and fps:
        frame_count = int(round(duration * fps))
    else:
        frame_count = 0

    return VideoInfo(
        width=width, height=height, fps=fps, frame_count=frame_count, duration_seconds=duration
    )


class VideoStreamReader:
    """Streams frames from a video file one at a time via OpenCV, without
    ever holding more than one frame in memory. Supports seeking to a start
    frame for checkpoint-based resume.
    """

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"video file not found: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self, start_frame: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, frame_bgr)`` starting at ``start_frame``.

        Seeking on HEVC streams remuxed with ``-c:v copy`` is only
        keyframe-accurate at the container level; OpenCV's FFmpeg backend
        compensates by decoding forward from the nearest keyframe, so the
        returned frame index is still correct, it just costs extra decode
        time proportional to the GOP size.
        """
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoStreamReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def parse_start_datetime(value: str) -> datetime:
    """Parse the ``video.start_datetime`` config string ("%Y-%m-%d %H:%M:%S")."""
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def frame_idx_to_timestamp(frame_idx: int, fps: float, start_dt: datetime) -> datetime:
    """Map a frame index to a wall-clock timestamp given the video's start time."""
    if fps <= 0:
        raise ValueError("fps must be > 0 to compute a timestamp")
    return start_dt + timedelta(seconds=frame_idx / fps)


def hour_bucket(ts: datetime) -> str:
    """Format a timestamp into an hourly bucket key, e.g. '2026-07-28 08:00'."""
    return ts.strftime("%Y-%m-%d %H:00")

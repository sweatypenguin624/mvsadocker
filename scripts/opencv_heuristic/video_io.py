"""Local video resolution: optional DAV->MP4 conversion, and metadata reads.

Replaces the Colab notebook's Drive-mount + fixed-source-path copy step --
here the caller just passes a local video path (any container OpenCV/ffmpeg
can handle). A .dav input is transparently converted to .mp4 next to the
requested output directory; anything else is used as-is.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def mp4_is_playable(path: str) -> bool:
    """Quick check that OpenCV can actually open and read a frame from path."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def convert_dav_to_mp4(dav_path: Path, mp4_path: Path) -> Path:
    """Convert dav_path to MP4 via FFmpeg. Never modifies the original.
    Reuses an existing MP4 only if it is confirmed playable by OpenCV;
    otherwise deletes the stale/corrupt file and re-converts.
    """
    if not dav_path.exists() or dav_path.stat().st_size == 0:
        raise RuntimeError(f"Cannot convert: source DAV missing or empty at {dav_path}.")
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        if mp4_is_playable(str(mp4_path)):
            print(f"Found existing converted MP4 at {mp4_path}, verified playable. Reusing it.")
            return mp4_path
        print(f"Existing MP4 at {mp4_path} exists but is NOT playable (stale/corrupt "
              "from a prior interrupted run). Deleting it and re-converting.")
        mp4_path.unlink()

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting DAV -> MP4:\n  in:  {dav_path}\n  out: {mp4_path}")
    cmd = ["ffmpeg", "-y", "-i", str(dav_path), "-c:v", "libx264", "-preset", "fast",
           "-crf", "18", "-an", str(mp4_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("---- FFmpeg stderr (tail) ----")
    print("\n".join(proc.stderr.splitlines()[-40:]))
    if proc.returncode != 0:
        if mp4_path.exists():
            mp4_path.unlink()
        raise RuntimeError(
            "FFmpeg failed to convert the DAV file to MP4 (possibly an unsupported "
            "proprietary codec). Full stderr above. Aborting -- will NOT silently "
            "fall back to another video."
        )
    if not mp4_is_playable(str(mp4_path)):
        if mp4_path.exists():
            mp4_path.unlink()
        raise RuntimeError(
            "FFmpeg reported success (exit code 0) but the output MP4 is not "
            "readable by OpenCV -- likely a codec/container mismatch, or FFmpeg "
            "silently produced an unreadable file. Full stderr above."
        )
    print(f"Conversion complete and verified playable: {mp4_path} ({mp4_path.stat().st_size} bytes)")
    return mp4_path


def resolve_video(video_path: Path, work_dir: Path) -> Path:
    """Return a path OpenCV can open, converting .dav inputs to .mp4 first."""
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path}")
    if video_path.suffix.lower() != ".dav":
        return video_path
    mp4_path = work_dir / (video_path.stem + ".mp4")
    return convert_dav_to_mp4(video_path, mp4_path)


def read_video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps is None or fps <= 0:
        raise RuntimeError(f"Video FPS is invalid ({fps}).")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video resolution is invalid ({width}x{height}).")
    duration_seconds = (frame_count / fps) if frame_count > 0 else 0.0
    print("\nVideo:")
    print(f"FPS: {fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {timedelta(seconds=int(duration_seconds))}")
    return VideoMeta(str(video_path), fps, frame_count, width, height, duration_seconds)

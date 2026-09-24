"""Video sampling, scoring and best-frame selection.

``VideoFrameAnalyzer`` decodes a sample of frames from a video and scores
each with ``FrameQualityScorer``. ``select_best_frames`` then picks the
top-K by score subject to a diversity constraint, so the result is not
just K near-duplicate frames from one flattering half-second -- a plain
top-K-by-score pick tends to cluster around whichever single moment had
the sharpest, best-lit frame, since neighbouring frames score almost
identically.

Frames are scored at a reduced resolution for speed and to make the
sharpness/contrast reference values resolution-independent; use
``extract_frames``/``save_selected_frames`` to pull the corresponding
full-resolution frames back out for output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from frame_quality import FrameQuality, FrameQualityScorer

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    index: int              # frame index in the source video
    timestamp: float        # seconds from the start of the video (best effort)
    phash: int              # 64-bit average-hash, for near-duplicate detection
    quality: FrameQuality

    @property
    def score(self) -> float:
        return self.quality.score

    def as_dict(self) -> dict:
        d = {"index": self.index, "timestamp": round(self.timestamp, 3)}
        d.update(self.quality.as_dict())
        return d


def _phash64(gray: np.ndarray) -> int:
    """Cheap 8x8 average-hash used only to detect near-duplicate frames
    between candidates -- not part of the quality score itself."""
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small.flatten() > avg).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class VideoFrameAnalyzer:
    """Samples and scores frames from a video file."""

    def __init__(
        self,
        scorer: Optional[FrameQualityScorer] = None,
        resize_width: Optional[int] = 960,
    ):
        self.scorer = scorer or FrameQualityScorer()
        # Frames are downscaled to this width before scoring, for speed
        # and so sharpness/contrast references stay meaningful regardless
        # of the source resolution. None disables downscaling.
        self.resize_width = resize_width

    def analyze(
        self,
        video_path: str,
        stride: Optional[int] = None,
        max_samples: int = 300,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[FrameResult]:
        """Decode and score a sample of frames from ``video_path``.

        ``stride`` scores every Nth frame; if not given it is derived
        from ``max_samples`` so long videos stay cheap to analyze.

        Samples are pulled via ``CAP_PROP_POS_FRAMES`` seeks rather than
        sequentially decoding and discarding every frame in between --
        for a large stride on a long video, decoding every frame to reach
        the next sample is dramatically slower (an hour-long 1080p file
        can take tens of minutes to decode sequentially vs. well under a
        minute to seek-sample). Some codecs can only seek to the nearest
        keyframe, so the frame actually decoded may land a little off the
        requested index; the reported index/timestamp reflect what was
        actually scored, not the nominal target.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if stride is None:
                stride = self._auto_stride(frame_count, max_samples)
            stride = max(1, stride)

            start_frame = int(start_time * fps) if (fps > 0 and start_time > 0) else 0
            if end_time and fps > 0:
                end_frame = int(end_time * fps)
            elif frame_count > 0:
                end_frame = frame_count
            else:
                end_frame = None  # unknown length; rely on read() failure to stop

            results: List[FrameResult] = []
            target = start_frame
            consecutive_failures = 0
            max_consecutive_failures = 10

            while end_frame is None or target < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    target += stride
                    continue
                consecutive_failures = 0

                # After a successful read(), POS_FRAMES is the index of the
                # *next* frame to decode, so the frame we just got is one less.
                reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                actual_index = reported if reported >= 0 else target

                try:
                    results.append(self._score_frame(frame, actual_index, fps))
                except Exception:
                    logger.warning("failed to score frame %d, skipping", actual_index, exc_info=True)

                target += stride

            if not results:
                raise RuntimeError(f"no frames could be decoded/scored from {video_path}")

            return results
        finally:
            cap.release()

    def _score_frame(self, frame_bgr: np.ndarray, index: int, fps: float) -> FrameResult:
        scored_frame = frame_bgr
        if self.resize_width and frame_bgr.shape[1] > self.resize_width:
            h, w = frame_bgr.shape[:2]
            new_h = max(1, int(round(h * (self.resize_width / w))))
            scored_frame = cv2.resize(
                frame_bgr, (self.resize_width, new_h), interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(scored_frame, cv2.COLOR_BGR2GRAY)
        phash = _phash64(gray)
        quality = self.scorer.score(scored_frame)
        timestamp = index / fps if fps > 0 else float(index)

        return FrameResult(index=index, timestamp=timestamp, phash=phash, quality=quality)

    @staticmethod
    def _auto_stride(frame_count: int, max_samples: int) -> int:
        if frame_count <= 0 or max_samples <= 0:
            return 1
        return max(1, frame_count // max_samples)


def select_best_frames(
    results: Sequence[FrameResult],
    top_k: int = 5,
    min_score: float = 0.0,
    min_gap_seconds: float = 1.0,
    dedupe_hamming_threshold: int = 6,
) -> List[FrameResult]:
    """Greedily pick the top-K highest-scoring frames, skipping any
    candidate that is too close in time to, or visually near-identical
    to, a frame already selected.

    Without this, the top-K by raw score alone tends to be K consecutive
    or near-consecutive frames from a single moment, since neighbouring
    frames in a video are nearly identical and score almost the same.
    """
    if top_k <= 0:
        return []

    candidates = sorted(
        (r for r in results if r.score >= min_score),
        key=lambda r: r.score,
        reverse=True,
    )

    selected: List[FrameResult] = []
    for candidate in candidates:
        if len(selected) >= top_k:
            break

        if min_gap_seconds > 0 and any(
            abs(candidate.timestamp - s.timestamp) < min_gap_seconds for s in selected
        ):
            continue

        if dedupe_hamming_threshold >= 0 and any(
            _hamming(candidate.phash, s.phash) <= dedupe_hamming_threshold for s in selected
        ):
            continue

        selected.append(candidate)

    return selected


def extract_frames(video_path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Re-open ``video_path`` and pull out the full-resolution frames at
    ``indices`` by seeking directly to each one. ``indices`` is normally
    a small set of already-selected frames that can be far apart in a
    long video, so seeking per-frame is far cheaper than one sequential
    pass through everything in between."""
    wanted = sorted(set(indices))
    if not wanted:
        return {}

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    found: Dict[int, np.ndarray] = {}
    try:
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                found[index] = frame
    finally:
        cap.release()

    missing = set(wanted) - found.keys()
    if missing:
        logger.warning("could not re-extract %d frame(s): %s", len(missing), sorted(missing))

    return found


def save_selected_frames(
    video_path: str,
    selected: Sequence[FrameResult],
    out_dir: str,
    prefix: str = "frame",
    image_ext: str = ".jpg",
) -> List[Path]:
    """Extract full-resolution frames for ``selected`` and write them to
    ``out_dir``, named with their rank, source frame index and score so
    the files sort in quality order and are traceable back to the video."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path, [r.index for r in selected])

    written: List[Path] = []
    for rank, result in enumerate(
        sorted(selected, key=lambda r: r.score, reverse=True), start=1
    ):
        frame = frames.get(result.index)
        if frame is None:
            continue
        filename = f"{prefix}_{rank:02d}_idx{result.index:07d}_score{result.score:.1f}{image_ext}"
        file_path = out_path / filename
        cv2.imwrite(str(file_path), frame)
        written.append(file_path)

    return written

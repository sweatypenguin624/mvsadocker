"""Line-crossing counting and 15-minute bucketing.

A vehicle is counted at most once, on the frame its ground-contact point
(bottom-centre of the box) crosses the calibrated counting line. Two
guards keep a box jittering on the line from registering repeatedly:

  - the crossing must be a genuine segment intersection with the FINITE
    line, not merely a sign change against the infinite line through it;
  - the vehicle must be observed on each side for a minimum number of
    frames, so a one-frame wobble across the line is not a crossing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from stage1_config import CountingConfig
from stage1_geometry import ground_point, scale_line, segment_intersects, side_of_line
from stage1_identity import VehicleIdentity

logger = logging.getLogger("mvsa.traffic_stage1")


@dataclass
class CountedVehicle:
    vehicle_id: int
    raw_track_ids: List[int]
    crossing_frame: int
    crossing_time: datetime
    interval: str
    direction: str
    frames_seen: int
    first_frame: int
    last_frame: int
    best_bbox: List[float]
    mean_conf: float
    merge_reasons: List[str]


def resolve_line(config: CountingConfig, frame_w: int, frame_h: int) -> Tuple[List[List[float]], bool]:
    """Return (line in frame pixels, calibrated?)."""
    if config.line:
        return scale_line(config.line, config.reference_width, config.reference_height, frame_w, frame_h), True
    y = frame_h * config.fallback_line_y_frac
    logger.warning(
        "No counting line calibrated for this camera -- falling back to a horizontal line at "
        "y=%.0f (%.0f%% of frame height). Counts are directionally useful but NOT calibrated; "
        "run calibrate_line.py to fix this.",
        y,
        config.fallback_line_y_frac * 100,
    )
    return [[0.0, y], [float(frame_w), y]], False


def interval_label(ts: datetime, minutes: int) -> str:
    """Floor a timestamp to its N-minute window, labelled by its start."""
    floored = ts.replace(
        minute=(ts.minute // minutes) * minutes, second=0, microsecond=0
    )
    return floored.strftime("%Y-%m-%d %H:%M")


def _run_before(sides: Sequence[float], i: int) -> int:
    """Consecutive frames ending at i-1 that share its side of the line."""
    positive = sides[i - 1] > 0
    n = 0
    for k in range(i - 1, -1, -1):
        if (sides[k] > 0) != positive:
            break
        n += 1
    return n


def _run_after(sides: Sequence[float], i: int) -> int:
    """Consecutive frames starting at i that share its side of the line."""
    positive = sides[i] > 0
    n = 0
    for k in range(i, len(sides)):
        if (sides[k] > 0) != positive:
            break
        n += 1
    return n


def _direction(before: Sequence[float], after: Sequence[float]) -> str:
    dx, dy = after[0] - before[0], after[1] - before[1]
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def count_vehicles(
    identities: List[VehicleIdentity],
    config: CountingConfig,
    frame_w: int,
    frame_h: int,
    frame_time: "callable",
) -> Tuple[List[CountedVehicle], Dict[str, int], List[List[float]], bool]:
    """Count each identity at most once. ``frame_time(frame_idx)`` -> datetime."""
    line, calibrated = resolve_line(config, frame_w, frame_h)
    p1, p2 = line

    counted: List[CountedVehicle] = []
    rejected: Dict[str, int] = {"too_few_frames": 0, "never_crossed": 0, "insufficient_dwell": 0}

    for identity in identities:
        obs = identity.observations
        if len(obs) < config.min_track_frames:
            rejected["too_few_frames"] += 1
            continue

        points = [ground_point(o.bbox) for o in obs]
        sides = [side_of_line(p, p1, p2) for p in points]

        crossing_at: Optional[int] = None
        for i in range(1, len(points)):
            if sides[i - 1] == 0 and sides[i] == 0:
                continue
            if (sides[i - 1] > 0) == (sides[i] > 0):
                continue
            if not segment_intersects(points[i - 1], points[i], p1, p2):
                continue
            # Confirm real dwell IMMEDIATELY either side of the crossing.
            # Counting matching frames anywhere in the track would pass a
            # box oscillating across the line, since such a track has
            # plenty of frames on both sides -- just never consecutively.
            if _run_before(sides, i) < config.min_frames_each_side:
                continue
            if _run_after(sides, i) < config.min_frames_each_side:
                continue
            crossing_at = i
            break

        if crossing_at is None:
            # Distinguish "never went near the line" from "wobbled on it",
            # so a badly placed line is visible in the summary.
            any_sign_change = any((sides[i - 1] > 0) != (sides[i] > 0) for i in range(1, len(sides)))
            rejected["insufficient_dwell" if any_sign_change else "never_crossed"] += 1
            continue

        crossing_frame = obs[crossing_at].frame_idx
        ts = frame_time(crossing_frame)
        best = max(obs, key=lambda o: o.conf)
        counted.append(
            CountedVehicle(
                vehicle_id=identity.vehicle_id,
                raw_track_ids=identity.raw_track_ids,
                crossing_frame=crossing_frame,
                crossing_time=ts,
                interval=interval_label(ts, config.interval_minutes),
                direction=_direction(points[crossing_at - 1], points[crossing_at]),
                frames_seen=len(obs),
                first_frame=identity.first_frame,
                last_frame=identity.last_frame,
                best_bbox=[round(v, 1) for v in best.bbox],
                mean_conf=round(sum(o.conf for o in obs) / len(obs), 4),
                merge_reasons=identity.merge_reasons,
            )
        )

    counted.sort(key=lambda c: c.crossing_frame)
    logger.info(
        "Counted %d vehicles (rejected: %s)",
        len(counted),
        ", ".join(f"{k}={v}" for k, v in rejected.items()),
    )
    return counted, rejected, line, calibrated


def build_intervals(
    counted: List[CountedVehicle],
    config: CountingConfig,
    start: Optional[datetime],
    end: Optional[datetime],
) -> List[Dict]:
    """Dense 15-min windows covering the whole video, zeros included.

    Stage 1 cannot fill in class counts -- that is Stage 2's job -- so each
    window reports only the total and the vehicle ids inside it, which is
    exactly the handoff Stage 2 needs.
    """
    buckets: Dict[str, List[int]] = {}
    for c in counted:
        buckets.setdefault(c.interval, []).append(c.vehicle_id)

    labels: List[str] = []
    if start and end:
        cursor = start.replace(minute=(start.minute // config.interval_minutes) * config.interval_minutes,
                               second=0, microsecond=0)
        while cursor <= end:
            labels.append(cursor.strftime("%Y-%m-%d %H:%M"))
            cursor += timedelta(minutes=config.interval_minutes)
    for label in buckets:
        if label not in labels:
            labels.append(label)
    labels.sort()

    return [
        {
            "interval_start": label,
            "interval_minutes": config.interval_minutes,
            "total_vehicles": len(buckets.get(label, [])),
            "vehicle_ids": sorted(buckets.get(label, [])),
        }
        for label in labels
    ]

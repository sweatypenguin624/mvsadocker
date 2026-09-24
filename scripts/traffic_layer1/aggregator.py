"""Run-level aggregation: per-class counts, directional counts, configurable
time-bucket aggregation, and the final run summary (both the dict written to
summary.json and the human-readable stdout block from spec section 15).

``time_bucket_key`` generalizes scripts/video_utils.py::hour_bucket to an
arbitrary bucket size (config: aggregation.time_bucket_minutes, default 60,
which reproduces hour_bucket's exact output format). Kept local to this
module rather than added to the shared video_utils.py, which three other
pipelines and their tests already depend on -- isolating this change avoids
touching code outside Layer 1's own footprint.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from movement_direction import ALL_LABELS  # noqa: E402

from layer1_models import TrackState  # noqa: E402
from taxonomy import BROAD_CLASSES  # noqa: E402


def time_bucket_key(ts: datetime, minutes: int) -> str:
    """Floor ``ts`` to the start of its ``minutes``-wide bucket, e.g. with
    minutes=60: '2026-07-28 08:00' (identical to hour_bucket's format);
    minutes=15: '2026-07-28 08:15'.
    """
    if minutes <= 0:
        raise ValueError("time_bucket_minutes must be > 0")
    floored_minute = (ts.minute // minutes) * minutes
    bucketed = ts.replace(minute=floored_minute, second=0, microsecond=0)
    return bucketed.strftime("%Y-%m-%d %H:%M")


def build_class_counts(counted: List[TrackState]) -> Dict[str, int]:
    counts = {c: 0 for c in BROAD_CLASSES}
    for state in counted:
        counts[state.final_class] = counts.get(state.final_class, 0) + 1
    return counts


def build_direction_counts(counted: List[TrackState]) -> Dict[str, Dict[str, int]]:
    """{class: {direction_label: count, ..., "total": n}} for every class
    that had at least one counted track (classes with zero counted tracks
    are omitted rather than padded with an all-zero row).
    """
    per_class: Dict[str, Dict[str, int]] = {}
    for state in counted:
        row = per_class.setdefault(state.final_class, {label: 0 for label in ALL_LABELS})
        label = state.direction_label or "indeterminate"
        row[label] = row.get(label, 0) + 1
    for row in per_class.values():
        row["total"] = sum(row[label] for label in ALL_LABELS)
    return per_class


def build_time_buckets(counted: List[TrackState], minutes: int) -> Dict[str, Dict[str, int]]:
    """{bucket_key: {class: count, ..., "total": n}}, using each track's
    entered_roi_time (the moment it was actually counted) as its bucket
    timestamp, not first/last seen.
    """
    buckets: Dict[str, Dict[str, int]] = {}
    for state in counted:
        if state.entered_roi_time is None:
            continue
        key = time_bucket_key(state.entered_roi_time, minutes)
        row = buckets.setdefault(key, {})
        row[state.final_class] = row.get(state.final_class, 0) + 1
    for row in buckets.values():
        row["total"] = sum(v for k, v in row.items() if k != "total")
    return buckets


def build_direction_totals(counted: List[TrackState]) -> Dict[str, int]:
    totals = {label: 0 for label in ALL_LABELS}
    for state in counted:
        label = state.direction_label or "indeterminate"
        totals[label] = totals.get(label, 0) + 1
    return totals


def format_run_summary_text(summary: dict) -> str:
    """The human-readable stdout block from spec section 15."""
    line = "=" * 60
    lines = [line, "TRAFFIC ANALYSIS RUN SUMMARY", line, ""]

    inp = summary["input"]
    lines += [
        "Input:",
        f"  video: {inp['video']}",
        f"  duration: {inp['duration']}",
        f"  fps: {inp['fps']}",
        f"  frames processed: {inp['frames_processed']:,}",
        "",
    ]

    det = summary["detection"]
    lines += [
        "Detection:",
        f"  total detections: {det['total_detections']:,}",
        f"  unique tracks: {det['unique_tracks']:,}",
        "",
    ]

    lines += ["Counted objects:", "-" * 60, f"{'Class':<28}{'Count':>10}", "-" * 60]
    total_counted = 0
    for cls, count in summary["class_counts"].items():
        lines.append(f"{cls:<28}{count:>10,}")
        total_counted += count
    lines += ["-" * 60, f"{'TOTAL':<28}{total_counted:>10,}", ""]

    trk = summary["tracking"]
    lines += [
        "Tracking:",
        f"  completed tracks: {trk['completed_tracks']:,}",
        f"  incomplete tracks: {trk['incomplete_tracks']:,}",
        f"  duplicate count prevented: {'YES' if trk['duplicate_count_prevented'] else 'NO'}",
        "",
    ]

    lines.append("Direction:")
    for label, count in summary["direction_totals"].items():
        lines.append(f"  {label.replace('_', ' '):<20} {count:,}")
    lines += ["", line, "RUN COMPLETE", line]

    return "\n".join(lines)

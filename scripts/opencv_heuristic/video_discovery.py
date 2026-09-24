"""Video discovery + camera name / real-world start-time inference for the
production pipeline.

Filenames in this project follow a couple of known patterns (see actual
files under videos/actual_test/):
  - "08.00.00-09.00.00.mp4"          -- HH.MM.SS-HH.MM.SS, start-end of hour
  - "20260720_090553_tp00075.mp4"    -- YYYYMMDD_HHMMSS_<tag>

Both are parsed for a wall-clock start time so 15-minute windows can be real
times ("09:00-09:15") instead of just video-relative offsets. Neither
pattern carries a date for the HH.MM.SS-HH.MM.SS form, so that one borrows
the file's mtime date -- if a file was copied/moved after recording, that
date can be wrong. If no pattern matches at all, falls back to full mtime.
Either fallback is logged loudly (see run_production.py) since it directly
affects which window a bus gets counted into; --video-start-times in the
config overrides per-file when the auto-detected date is wrong.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_RANGE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})")
_COMPACT_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def discover_videos(folder: Path, globs: list) -> list:
    """All files under folder (recursive) matching any of globs, deduped and sorted."""
    seen = set()
    out = []
    for pattern in globs:
        for p in sorted(folder.rglob(pattern)):
            if p in seen or not p.is_file() or p.name.startswith("."):
                continue
            seen.add(p)
            out.append(p)
    return sorted(out)


def derive_camera_name(video_path: Path, configured: str | None) -> str:
    """Explicit config value wins; otherwise the containing folder name is
    treated as the camera identity (one folder per camera is the expected
    layout for mode='folder')."""
    if configured:
        return configured
    return video_path.parent.name or "unknown_camera"


def parse_video_start_time(video_path: Path, override: str | None = None):
    """Best-effort real-world start datetime for a video file.

    Returns (datetime, source) where source is one of:
      "override"        -- from run_config.video_start_times
      "range_pattern"    -- "HH.MM.SS-HH.MM.SS..." filename, date borrowed from mtime
      "compact_pattern"  -- "YYYYMMDD_HHMMSS..." filename, fully self-contained
      "mtime"            -- no pattern matched; caller should warn, this is unreliable
    """
    if override:
        return datetime.fromisoformat(override), "override"

    name = video_path.name
    m = _RANGE_RE.search(name)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mtime_date = datetime.fromtimestamp(video_path.stat().st_mtime).date()
        return (
            datetime.combine(mtime_date, datetime.min.time()).replace(hour=h, minute=mi, second=s),
            "range_pattern",
        )

    m = _COMPACT_RE.search(name)
    if m:
        y, mo, d, h, mi, s = (int(g) for g in m.groups())
        return datetime(y, mo, d, h, mi, s), "compact_pattern"

    return datetime.fromtimestamp(video_path.stat().st_mtime), "mtime"

"""Layer 1 track-level data structures.

``Detection`` and ``BBox`` are reused as-is from scripts/models.py (the
repo-wide shared dataclass module) -- nothing new is defined for a single
per-frame detection here, only the track-level state built up across frames,
which none of the existing pipelines have in this shape (broad-bucket class
votes + bounded trajectory history + direction + crop bookkeeping all in one
place).

History lists are bounded (``collections.deque(maxlen=...)``) rather than
growing for the life of a track, per the spec's streaming/performance
requirement -- an hour-long track must not accumulate an unbounded list of
every frame it was ever seen in.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models import BBox  # noqa: E402  (shared repo-wide dataclass module)

Point = Tuple[float, float]


@dataclass
class SourcedDetection:
    """One frame's detection from one of Layer 1's two detector sources,
    tagged with which model produced it and that model's native class name
    (resolved from the model's own ``.names``, never assumed -- see
    detector.py). ``bucket`` is filled in by the pipeline once taxonomy.py
    maps ``native_class`` to a Layer 1 broad class.
    """

    source: str  # "uvh26" or "coco"
    native_class: str
    bbox: BBox
    conf: float
    raw_track_id: Optional[int]
    bucket: Optional[str] = None

# How many recent (bbox, center, confidence) samples a track keeps for
# trajectory/direction fitting and diagnostics. Direction classification
# only needs the shape of the trajectory, not every single frame, and this
# bound is what keeps memory flat for very long-lived tracks (e.g. a parked
# vehicle sitting in frame for the whole video).
TRACK_HISTORY_MAXLEN = 300


@dataclass
class TrackState:
    """Per-track state, keyed by (source, raw_track_id) in the tracker but
    exposed under one synthetic, globally-unique ``track_id`` in all output
    (see layer1_tracker.py -- ByteTrack ID spaces from the two detector sources are
    independent and would otherwise collide).
    """

    track_id: int
    source: str  # "uvh26" or "coco"
    raw_track_id: int

    first_seen_frame: int
    first_seen_time: datetime
    last_seen_frame: int
    last_seen_time: datetime
    frames_seen: int = 0

    # bucket name -> summed confidence, accumulated every frame this track
    # was observed. classifier.py::vote_class reduces this to one label.
    class_votes: Dict[str, float] = field(default_factory=dict)

    confidence_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY_MAXLEN)
    )
    bbox_history: Deque[BBox] = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))
    # Bottom-center ("foot point") of each observed bbox, in frame pixel
    # space -- the same point convention roi.py/movement_direction.py use.
    center_history: Deque[Point] = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))

    counted: bool = False
    entered_roi_frame: Optional[int] = None
    entered_roi_time: Optional[datetime] = None
    # Final broad class, resolved once (at counting time) via
    # classifier.py::vote_class -- not recomputed per frame afterwards.
    final_class: Optional[str] = None
    direction_label: Optional[str] = None
    direction_displacement_px: float = 0.0

    def add_observation(
        self, frame_idx: int, timestamp: datetime, bucket: str, conf: float, bbox: BBox
    ) -> None:
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.frames_seen += 1
        self.class_votes[bucket] = self.class_votes.get(bucket, 0.0) + conf
        self.confidence_history.append(conf)
        self.bbox_history.append(bbox)
        x1, y1, x2, y2 = bbox
        self.center_history.append(((x1 + x2) / 2.0, y2))

    @property
    def mean_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def max_confidence(self) -> float:
        return max(self.confidence_history) if self.confidence_history else 0.0


@dataclass
class CountedRecord:
    """One finalized, counted track -- the record spec section 11 describes,
    and the row written to tracks.jsonl / counts.csv.
    """

    track_id: int
    source: str
    cls: str
    confidence: float
    direction: str
    first_seen: datetime
    last_seen: datetime
    counted_at: datetime
    frames_seen: int
    has_bus_crops: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "source": self.source,
            "class": self.cls,
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "first_seen": self.first_seen.isoformat(sep=" "),
            "last_seen": self.last_seen.isoformat(sep=" "),
            "counted_at": self.counted_at.isoformat(sep=" "),
            "frames_seen": self.frames_seen,
            "has_bus_crops": self.has_bus_crops,
        }

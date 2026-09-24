"""Pedestrian ROI-entry tracking.

Owns the mapping from raw per-frame Detections to "has this ByteTrack ID
entered the pedestrian ROI yet". A track is counted exactly once, the first
frame its bounding box's bottom-center point falls inside the ROI polygon.
If a person leaves frame and later returns, ByteTrack will normally assign a
brand-new track_id -- and that new ID is allowed to count again. No
cross-session re-identification is attempted, by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from models import Detection
from roi import PedestrianROI, bottom_center


@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int
    first_seen_time: datetime
    last_seen_frame: int
    last_seen_time: datetime
    frames_seen: int = 0
    counted: bool = False
    entered_roi_frame: Optional[int] = None
    rider_statuses: List[str] = field(default_factory=list)
    rider_overlap_ratios: List[float] = field(default_factory=list)
    demographic_samples_taken: int = 0
    frames_since_last_sample: int = 0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "first_seen_frame": self.first_seen_frame,
            "first_seen_time": self.first_seen_time.isoformat(),
            "last_seen_frame": self.last_seen_frame,
            "last_seen_time": self.last_seen_time.isoformat(),
            "frames_seen": self.frames_seen,
            "counted": self.counted,
            "entered_roi_frame": self.entered_roi_frame,
            "rider_statuses": self.rider_statuses,
            "rider_overlap_ratios": self.rider_overlap_ratios,
            "demographic_samples_taken": self.demographic_samples_taken,
            "frames_since_last_sample": self.frames_since_last_sample,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrackState":
        return TrackState(
            track_id=int(d["track_id"]),
            first_seen_frame=int(d["first_seen_frame"]),
            first_seen_time=datetime.fromisoformat(d["first_seen_time"]),
            last_seen_frame=int(d["last_seen_frame"]),
            last_seen_time=datetime.fromisoformat(d["last_seen_time"]),
            frames_seen=int(d.get("frames_seen", 0)),
            counted=bool(d.get("counted", False)),
            entered_roi_frame=d.get("entered_roi_frame"),
            rider_statuses=list(d.get("rider_statuses", [])),
            rider_overlap_ratios=list(d.get("rider_overlap_ratios", [])),
            demographic_samples_taken=int(d.get("demographic_samples_taken", 0)),
            frames_since_last_sample=int(d.get("frames_since_last_sample", 0)),
        )


class PedestrianTracker:
    """Tracks per-ByteTrack-ID state and detects ROI-entry events."""

    def __init__(self, roi: PedestrianROI, frame_width: int, frame_height: int):
        self.roi = roi
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracks: Dict[int, TrackState] = {}
        self.counted_ids: Set[int] = set()

    def update(
        self, frame_idx: int, timestamp: datetime, person_detections: List[Detection]
    ) -> List[int]:
        """Update track state with this frame's person detections.

        Returns the list of track_ids that newly entered the ROI on this
        frame (i.e. should be counted now).
        """
        newly_counted: List[int] = []

        for det in person_detections:
            if det.track_id is None:
                continue

            state = self.tracks.get(det.track_id)
            if state is None:
                state = TrackState(
                    track_id=det.track_id,
                    first_seen_frame=frame_idx,
                    first_seen_time=timestamp,
                    last_seen_frame=frame_idx,
                    last_seen_time=timestamp,
                )
                self.tracks[det.track_id] = state

            state.last_seen_frame = frame_idx
            state.last_seen_time = timestamp
            state.frames_seen += 1
            state.frames_since_last_sample += 1

            if not state.counted:
                foot_point = bottom_center(det.bbox)
                if self.roi.contains_point(foot_point, self.frame_width, self.frame_height):
                    state.counted = True
                    state.entered_roi_frame = frame_idx
                    self.counted_ids.add(det.track_id)
                    newly_counted.append(det.track_id)

        return newly_counted

    def restore(self, tracks: Dict[int, TrackState], counted_ids: Set[int]) -> None:
        """Restore state from a checkpoint (used by pipeline.py on resume)."""
        self.tracks = tracks
        self.counted_ids = counted_ids

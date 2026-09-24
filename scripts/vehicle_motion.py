"""Parked-vehicle detection for the rider filter.

scripts/rider_filter.py classifies a person as a "rider" purely from bbox
overlap with a same-frame vehicle detection -- it has no way to tell a
person riding a vehicle from a person merely standing or walking next to
one that's parked. On a street lined with parked two-wheelers (the common
case outside shops), that blind spot routes real pedestrians into
"rider"/"uncertain" and excludes them from the count.

This module tracks each vehicle-class ByteTrack ID's position over time and
flags one as "parked" once it has stayed within a small drift radius for a
sustained window -- long enough that a genuinely stopped-but-still-being-
ridden vehicle (e.g. waiting at a red light) is very unlikely to trigger it,
but short enough to catch a vehicle parked for most of the video. Once a
vehicle is flagged parked, the pipeline stops treating overlap with it as
rider evidence (see pipeline.py's frame loop).

Known tradeoff: a vehicle stopped continuously (same ByteTrack ID, no
occlusion) for longer than the configured window -- e.g. stationary in slow
traffic -- will eventually be flagged parked too, and a rider still astride
it at that point would lose their rider-overlap signal. This is a
deliberate bet that permanently-parked vehicles are the far more common
case; if a camera's traffic regularly stalls for this long, revisit
rider_filter.parked_vehicle_stationary_seconds for that camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from models import Detection


@dataclass
class VehicleMotionState:
    # Frame/position this vehicle's current "how long has it sat still"
    # streak started from. Reset forward whenever the vehicle drifts more
    # than max_drift_px from this anchor, so a vehicle that moves, then
    # parks, only starts accumulating "parked" time from when it stopped.
    anchor_frame_idx: int
    anchor_center: Tuple[float, float]
    last_frame_idx: int

    def to_dict(self) -> dict:
        return {
            "anchor_frame_idx": self.anchor_frame_idx,
            "anchor_center": list(self.anchor_center),
            "last_frame_idx": self.last_frame_idx,
        }

    @staticmethod
    def from_dict(d: dict) -> "VehicleMotionState":
        return VehicleMotionState(
            anchor_frame_idx=int(d["anchor_frame_idx"]),
            anchor_center=(float(d["anchor_center"][0]), float(d["anchor_center"][1])),
            last_frame_idx=int(d["last_frame_idx"]),
        )


class VehicleMotionTracker:
    """Tracks per-ByteTrack-ID vehicle position history and flags IDs that
    have been stationary for at least ``parked_window_frames``.
    """

    def __init__(self, parked_window_frames: int, max_drift_px: float):
        self.parked_window_frames = parked_window_frames
        self.max_drift_px = max_drift_px
        self.states: Dict[int, VehicleMotionState] = {}

    def update(self, frame_idx: int, vehicle_detections: List[Detection]) -> None:
        for det in vehicle_detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            state = self.states.get(det.track_id)
            if state is None:
                self.states[det.track_id] = VehicleMotionState(frame_idx, center, frame_idx)
                continue

            drift = ((center[0] - state.anchor_center[0]) ** 2 + (center[1] - state.anchor_center[1]) ** 2) ** 0.5
            if drift > self.max_drift_px:
                state.anchor_frame_idx = frame_idx
                state.anchor_center = center
            state.last_frame_idx = frame_idx

    def is_parked(self, track_id: int, frame_idx: int) -> bool:
        state = self.states.get(track_id)
        if state is None:
            return False
        return (frame_idx - state.anchor_frame_idx) >= self.parked_window_frames

    def to_checkpoint(self) -> dict:
        return {str(tid): s.to_dict() for tid, s in self.states.items()}

    def restore(self, data: dict) -> None:
        self.states = {int(tid): VehicleMotionState.from_dict(s) for tid, s in data.items()}

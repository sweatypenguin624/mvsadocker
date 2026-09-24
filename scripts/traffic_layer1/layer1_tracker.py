"""Layer 1 ROI-entry tracking across both detector sources at once.

Mirrors scripts/tracker.py's and scripts/vehicle_counting/vehicle_tracker.py's
identical design exactly: a track counts exactly once, on the first frame its
bounding box's bottom-center ("ground contact") point falls inside the
counting ROI (scripts/roi.py, reused unmodified). Generalized here to:

  - key tracks by (source, raw_track_id), since UVH-26's and the COCO
    person-detector's ByteTrack instances have independent ID spaces (a
    "track 5" from each is a different, unrelated physical object) --
    each gets its own synthetic, globally-unique ``track_id`` for output.
  - resolve a track's final broad class via classifier.py::vote_class
    (confidence-weighted majority vote over its whole observation history)
    at the moment it's counted, rather than fixing the class to whatever
    the detector reported on the entry frame alone (the vehicle pilot's
    current, simpler behaviour) -- this is the track-level classification
    the spec's section 7 asks for.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from roi import PedestrianROI, bottom_center  # noqa: E402

from classifier import vote_class  # noqa: E402
from layer1_models import SourcedDetection, TrackState  # noqa: E402

TrackKey = Tuple[str, int]  # (source, raw_track_id)


class Layer1Tracker:
    def __init__(
        self,
        roi: PedestrianROI,
        frame_width: int,
        frame_height: int,
    ):
        self.roi = roi
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.tracks: Dict[TrackKey, TrackState] = {}
        self._next_track_id = 1

    def update(
        self,
        frame_idx: int,
        timestamp: datetime,
        detections: List[SourcedDetection],
    ) -> List[TrackState]:
        """Update track state with this frame's merged detections from both
        sources. ``detections`` must already be bucket-tagged and threshold-
        filtered (see taxonomy.py::tag_and_filter -- called once by
        pipeline.py, since crops.py needs the same tagged list). Returns the
        TrackState entries that newly entered the ROI on this frame (i.e.
        should be counted now).
        """
        newly_counted: List[TrackState] = []

        for det in detections:
            if det.raw_track_id is None or det.bucket is None:
                continue

            key: TrackKey = (det.source, det.raw_track_id)
            state = self.tracks.get(key)
            if state is None:
                state = TrackState(
                    track_id=self._next_track_id,
                    source=det.source,
                    raw_track_id=det.raw_track_id,
                    first_seen_frame=frame_idx,
                    first_seen_time=timestamp,
                    last_seen_frame=frame_idx,
                    last_seen_time=timestamp,
                )
                self._next_track_id += 1
                self.tracks[key] = state

            state.add_observation(frame_idx, timestamp, det.bucket, det.conf, det.bbox)

            if not state.counted:
                foot_point = bottom_center(det.bbox)
                if self.roi.contains_point(foot_point, self.frame_width, self.frame_height):
                    state.counted = True
                    state.entered_roi_frame = frame_idx
                    state.entered_roi_time = timestamp
                    state.final_class = vote_class(state.class_votes)
                    newly_counted.append(state)

        return newly_counted

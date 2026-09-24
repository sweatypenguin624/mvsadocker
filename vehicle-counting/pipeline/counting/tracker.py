
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import Any
from enum import Enum

class TrackStateEnum(str, Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    COUNTED = "COUNTED"

logger = logging.getLogger("uvh_test.tracker")

@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    counted: bool = False
    state: TrackStateEnum = TrackStateEnum.NEW
    class_history: list = field(default_factory=list)
    conf_history: list = field(default_factory=list)
    previous_bottom_center: tuple = None
    max_lost_gap: int = 0
    total_lost_frames: int = 0
    creation_bbox: list = None
    last_bbox: list = None
    counting_bbox: list = None
    crossing_frame: int = -1
    crossing_direction: str = None
    not_counted_reason: str = "Never crossed counting line"
    best_crop: "np.ndarray" = None
    best_conf: float = 0.0

    @property
    def mean_confidence(self):
        return float(np.mean(self.conf_history)) if self.conf_history else 0.0

    @property
    def max_confidence(self):
        return float(np.max(self.conf_history)) if self.conf_history else 0.0

    def get_stable_class(self, class_names):
        if hasattr(self, "override_class") and self.override_class:
            return self.override_class
        if not self.class_history:
            return "Unknown"
        weighted_votes = {}
        for cls_id, conf in zip(self.class_history, self.conf_history):
            weighted_votes[cls_id] = weighted_votes.get(cls_id, 0.0) + conf
        best_cls = max(weighted_votes, key=weighted_votes.get)
        return class_names.get(best_cls, "Unknown")

class VehicleTracker:
    def __init__(self, min_track_frames: int):
        self.min_track_frames = min_track_frames
        self.tracks = {}
        
    def update(self, frame_idx: int, tracked_boxes: Any) -> list:
        active_tracks = []
        if tracked_boxes is None or len(tracked_boxes) == 0 or tracked_boxes.id is None:
            return active_tracks
            
        track_ids = tracked_boxes.id.cpu().numpy().astype(int)
        clss = tracked_boxes.cls.cpu().numpy().astype(int)
        confs = tracked_boxes.conf.cpu().numpy()
        xyxy = tracked_boxes.xyxy.cpu().numpy()
        
        for i, (tid, cls_id, conf, box) in enumerate(zip(track_ids, clss, confs, xyxy)):
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(
                    track_id=tid,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    creation_bbox=box.tolist() if hasattr(box, "tolist") else list(box)
                )
            
            state = self.tracks[tid]
            lost_gap = frame_idx - state.last_seen_frame - 1
            if lost_gap > 0:
                state.max_lost_gap = max(state.max_lost_gap, lost_gap)
                state.total_lost_frames += lost_gap
            state.last_seen_frame = frame_idx
            state.class_history.append(cls_id)
            state.conf_history.append(conf)
            state.last_bbox = box.tolist() if hasattr(box, "tolist") else list(box)
            
            frames_alive = frame_idx - state.first_seen_frame + 1
            if state.state == TrackStateEnum.NEW:
                if frames_alive >= self.min_track_frames:
                    state.state = TrackStateEnum.CONFIRMED
                    state.not_counted_reason = "Never crossed counting line"
                else:
                    state.not_counted_reason = "Unconfirmed (too short lifespan)"
            
            # Yield ALL active tracks (even NEW ones) so counter can track their early path
            active_tracks.append((tid, state, box))
                
        return active_tracks

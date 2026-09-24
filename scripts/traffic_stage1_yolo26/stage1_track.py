"""Tracker adapter: run Ultralytics' ByteTrack / BoT-SORT over OUR boxes.

Ultralytics' ``model.track()`` hides the tracker behind the predictor, so
detections go straight from NMS into the tracker with no chance to fuse
mis-split boxes first. Since fusion (stage1_fusion.py) is the whole point,
the tracker is instantiated directly and fed a duck-typed detection object
instead -- BYTETracker.update only ever touches ``.xywh``, ``.conf``,
``.cls`` and boolean-mask indexing.

Every detection is handed the same class id (0 = "vehicle"), so the
tracker's association is driven purely by motion and geometry and can
never break a track because the detector changed its mind about the label
mid-way -- a real source of ID switches in the previous pipeline.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
from ultralytics.trackers import BOTSORT, BYTETracker

from stage1_config import TrackerConfig

logger = logging.getLogger("mvsa.traffic_stage1")

VEHICLE_CLASS_ID = 0


class _Detections:
    """Minimal Results-like view over plain arrays."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.full(len(self.conf), VEHICLE_CLASS_ID, dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        b = self.xyxy
        return np.stack(
            [(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]],
            axis=1,
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Detections":
        return _Detections(self.xyxy[mask], self.conf[mask])


def _tracker_args(config: TrackerConfig) -> SimpleNamespace:
    return SimpleNamespace(
        tracker_type=config.tracker_type,
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
        fuse_score=config.fuse_score,
        gmc_method=config.gmc_method,
        proximity_thresh=config.proximity_thresh,
        appearance_thresh=config.appearance_thresh,
        with_reid=config.with_reid,
        model=config.reid_model,
    )


class VehicleTracker:
    """Wraps one tracker instance; returns (track_id, bbox, conf) per frame."""

    def __init__(self, config: TrackerConfig, frame_rate: int = 30):
        self.config = config
        args = _tracker_args(config)
        if config.tracker_type == "botsort":
            self.tracker = BOTSORT(args)
        elif config.tracker_type == "bytetrack":
            self.tracker = BYTETracker(args)
        else:
            raise ValueError(f"Unsupported tracker_type: {config.tracker_type}")
        logger.info(
            "Tracker ready: %s (buffer=%d frames, reid=%s)",
            config.tracker_type,
            config.track_buffer,
            config.with_reid,
        )

    def update(self, boxes: np.ndarray, confs: np.ndarray, frame: Optional[np.ndarray] = None):
        """Returns a list of (track_id, [x1,y1,x2,y2], conf) for this frame."""
        dets = _Detections(boxes, confs)
        out = self.tracker.update(dets, frame)
        results = []
        if out is None or len(out) == 0:
            return results
        for row in np.asarray(out):
            x1, y1, x2, y2, track_id, score = row[0], row[1], row[2], row[3], row[4], row[5]
            results.append((int(track_id), [float(x1), float(y1), float(x2), float(y2)], float(score)))
        return results

    def reset(self) -> None:
        self.tracker.reset()

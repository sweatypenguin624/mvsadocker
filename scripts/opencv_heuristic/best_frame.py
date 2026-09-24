"""Per-track best-frame selection for the production pipeline.

full_pipeline (mode_full_pipeline.py) reclassifies each track repeatedly
across its lifetime and fuses the results (TrackColorAccumulator). That's
robust but classifies a lot of mediocre crops. Production mode instead
picks exactly ONE frame per track -- the sharpest, largest, most confident,
least edge-clipped detection seen -- and classifies only that, once, when
the track's final window assignment is known. Only the current best crop
per active track is held in memory, never a frame history.
"""

from __future__ import annotations

import cv2


def sharpness_score(crop) -> float:
    """Laplacian variance on grayscale -- higher means sharper/more in-focus."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_detection(frame_shape, box, confidence, crop, cfg) -> float:
    """confidence * area_norm * sharpness_norm * edge_penalty, each in [0,1]
    (confidence and the norms are naturally bounded; edge_penalty is either
    1.0 or cfg.BEST_FRAME_EDGE_PENALTY)."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = float(w * h)
    area_norm = min(1.0, (box_area / frame_area) / cfg.BEST_FRAME_TARGET_AREA_FRAC) if frame_area > 0 else 0.0

    sharp_norm = min(1.0, sharpness_score(crop) / cfg.BEST_FRAME_SHARPNESS_REF)

    margin = cfg.BEST_FRAME_EDGE_MARGIN_PX
    touches_edge = x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin)
    edge_penalty = cfg.BEST_FRAME_EDGE_PENALTY if touches_edge else 1.0

    conf = float(confidence) if confidence is not None else 0.5
    return conf * area_norm * sharp_norm * edge_penalty


class BestFrameStore:
    """track_id -> best (score, crop, meta) seen so far. Only the winning
    crop is retained (copied out of the frame buffer, which gets reused by
    cv2.VideoCapture on the next read)."""

    def __init__(self):
        self._best = {}

    def offer(self, track_id, score, crop, meta):
        cur = self._best.get(track_id)
        if cur is None or score > cur[0]:
            self._best[track_id] = (score, crop.copy(), meta)

    def get(self, track_id):
        return self._best.get(track_id)

    def __contains__(self, track_id):
        return track_id in self._best

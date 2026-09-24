"""Tractor / Tractor Trolley relationship (spec sections 12-13).

Reduced-fidelity by design: Layer 1 only persists a handful of sparse,
top-scored (bbox, frame_idx) samples per track (see
scripts/traffic_layer1/crops.py), not a full per-frame trajectory, so this
evaluates persistence/proximity across whatever sparse frames a tractor
track and a candidate trolley track have in common -- not true continuous
tracking. Good enough to avoid a single-frame coincidence miscount (spec
section 13 explicitly warns against that), not a substitute for a real
frame-synchronized re-tracking pass.

NOTE: candidate trolley tracks are only usable if their Layer 1 broad
bucket is included in config/layer1_config.yaml's crops.target_classes (so
Layer 1 actually saved bbox samples for them) -- see
tractor_trolley.candidate_trolley_buckets in config/layer2_config.yaml. If
no candidate has saved crops, this always returns "no trolley found"
(Tractor, not Tractor Trolley) rather than guessing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from layer2_config import TractorTrolleyConfig
from layer2_models import GoodsTrack, RepresentativeFrame


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _shared_frame_links(
    a_frames: List[RepresentativeFrame], b_frames: List[RepresentativeFrame], max_dist: float
) -> int:
    """Count how many of A's sampled frames have a B frame within
    max_dist px of the same frame_idx (+/- a small tolerance) and within
    max_dist px spatially. Sparse-sample analogue of "persistent across
    multiple frames".
    """
    linked = 0
    for fa in a_frames:
        for fb in b_frames:
            if abs(fa.frame_idx - fb.frame_idx) > 5:
                continue
            ca, cb = _bbox_center(fa.bbox), _bbox_center(fb.bbox)
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            if dist <= max_dist:
                linked += 1
                break
    return linked


def find_attached_trolley(
    tractor_track: GoodsTrack,
    candidate_tracks: List[GoodsTrack],
    config: TractorTrolleyConfig,
) -> Optional[int]:
    """Return the track_id of the trolley most persistently spatially
    linked to tractor_track, or None if no candidate meets
    min_persistent_frames (spec section 13: single-frame proximity alone is
    not sufficient).
    """
    if not tractor_track.frames:
        return None

    best_track_id: Optional[int] = None
    best_links = 0
    for candidate in candidate_tracks:
        if candidate.track_id == tractor_track.track_id or not candidate.frames:
            continue
        links = _shared_frame_links(tractor_track.frames, candidate.frames, config.max_link_distance_px)
        if links >= config.min_persistent_frames and links > best_links:
            best_links = links
            best_track_id = candidate.track_id

    return best_track_id

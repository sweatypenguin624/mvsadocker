"""Unique-vehicle, single-best-frame selection for training data.

ByteTrack already gives every physical vehicle one stable track_id across
however many frames it's visible for (that's what makes counting correct --
one count per track, not per detection). The same identity guarantee makes
it the right unit for training data too: a track's frames are all crops of
the SAME vehicle, so keeping more than one is redundant near-duplicates,
not added diversity. This module picks exactly one frame per track -- the
one with the highest raw detector confidence (not the composite quality
score crops.py/goods_crops.py use for the *default* best.jpg, which also
weighs box size and blur) -- for the training dataset. Counting itself
needs no change: one track = one count is already how counter.py works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from layer2_models import GoodsTrack, RepresentativeFrame


def select_best_confidence_frame(track: GoodsTrack) -> Optional[RepresentativeFrame]:
    """The single frame with the highest raw detector confidence among a
    track's saved frames, or None if the track has no usable frames.
    """
    if not track.frames:
        return None
    return max(track.frames, key=lambda f: f.detector_conf)


def build_unique_vehicle_manifest(tracks: List[GoodsTrack]) -> List[Dict]:
    """One row per unique vehicle (track) with its single chosen frame --
    the manifest that drives both the "one frame per vehicle" training
    export and a sanity count of unique vehicles vs. raw crop volume.
    """
    rows = []
    for track in tracks:
        best = select_best_confidence_frame(track)
        rows.append({
            "track_id": track.track_id,
            "layer1_class": track.layer1_class,
            "total_frames_available": len(track.frames),
            "chosen_frame": best.file if best else None,
            "chosen_frame_confidence": round(best.detector_conf, 4) if best else None,
            "chosen_frame_path": str(best.path) if best else None,
        })
    return rows

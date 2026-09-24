"""Raw tracker IDs -> physical vehicle identities.

Run as a post-pass over the completed track table, not online, because
both corrections it makes need evidence from frames later than the one
where the tracker went wrong:

  (a) SPATIAL dedup -- two tracks alive simultaneously on one object.
      Whether two boxes are really one vehicle is only clear once you can
      see they stayed locked together for many frames rather than briefly
      overlapping as one vehicle passed another.

  (b) TEMPORAL stitching -- one vehicle split into track A then track B by
      an ID switch (occlusion, a missed detection run). You can only know
      B continues A after B exists.

The previous pipeline had NEITHER: it keyed vehicles purely on the raw
tracker id, so any tracker mistake became a counting mistake with nothing
downstream to catch it. Both corrections here only ever MERGE identities,
so they can reduce over-counting but can never delete a vehicle that was
detected -- the "no vehicle is missed" side is owned by the detector's low
confidence floor and the tracker's long buffer instead.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from stage1_appearance import similarity as appearance_similarity
from stage1_config import IdentityConfig
from stage1_geometry import iom, iou

logger = logging.getLogger("mvsa.traffic_stage1")


@dataclass
class Observation:
    frame_idx: int
    bbox: List[float]
    conf: float


@dataclass
class RawTrack:
    track_id: int
    observations: List[Observation] = field(default_factory=list)
    # Mean HSV fingerprint (stage1_appearance), or None when the vehicle
    # was never big enough on screen to sample reliably.
    appearance: Optional["np.ndarray"] = None

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_idx

    def box_at(self, frame_idx: int) -> Optional[List[float]]:
        for obs in self.observations:
            if obs.frame_idx == frame_idx:
                return obs.bbox
        return None

    def velocity(self, window: int = 5) -> Tuple[float, float]:
        """Mean per-frame centre displacement over the last ``window`` obs."""
        obs = self.observations[-(window + 1):]
        if len(obs) < 2:
            return 0.0, 0.0
        span = obs[-1].frame_idx - obs[0].frame_idx
        if span <= 0:
            return 0.0, 0.0
        c0 = _centre(obs[0].bbox)
        c1 = _centre(obs[-1].bbox)
        return (c1[0] - c0[0]) / span, (c1[1] - c0[1]) / span

    def head_velocity(self, window: int = 5) -> Tuple[float, float]:
        obs = self.observations[: window + 1]
        if len(obs) < 2:
            return 0.0, 0.0
        span = obs[-1].frame_idx - obs[0].frame_idx
        if span <= 0:
            return 0.0, 0.0
        c0 = _centre(obs[0].bbox)
        c1 = _centre(obs[-1].bbox)
        return (c1[0] - c0[0]) / span, (c1[1] - c0[1]) / span


def _centre(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box: Sequence[float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


class _DisjointSet:
    def __init__(self, keys):
        self._parent = {k: k for k in keys}

    def find(self, k):
        while self._parent[k] != k:
            self._parent[k] = self._parent[self._parent[k]]
            k = self._parent[k]
        return k

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = defaultdict(list)
        for k in self._parent:
            out[self.find(k)].append(k)
        return out


@dataclass
class VehicleIdentity:
    """One physical vehicle: the merged timeline of >=1 raw tracks."""

    vehicle_id: int
    raw_track_ids: List[int]
    observations: List[Observation]
    merge_reasons: List[str] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_idx

    @property
    def frames_seen(self) -> int:
        return len(self.observations)


def _spatial_pairs(tracks: Dict[int, RawTrack], cfg: IdentityConfig) -> List[Tuple[int, int, str]]:
    """Find track pairs that are two boxes on ONE object."""
    if not cfg.spatial_enabled:
        return []

    by_frame: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    for tid, track in tracks.items():
        for obs in track.observations:
            by_frame[obs.frame_idx].append((tid, obs.bbox))

    overlap_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for entries in by_frame.values():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (ta, ba), (tb, bb) = entries[i], entries[j]
                if iom(ba, bb) >= cfg.spatial_iom_thresh:
                    overlap_counts[(min(ta, tb), max(ta, tb))] += 1

    pairs = []
    for (ta, tb), count in overlap_counts.items():
        if count < cfg.spatial_min_overlap_frames:
            continue
        # The overlap must dominate the SHORTER track's life. One vehicle
        # briefly passing behind another produces a few overlapping frames
        # out of a long life; a mis-split box overlaps for essentially all
        # of its existence.
        shorter_life = min(len(tracks[ta].observations), len(tracks[tb].observations))
        if shorter_life <= 0:
            continue
        if count / shorter_life >= cfg.spatial_min_overlap_ratio:
            pairs.append((ta, tb, f"spatial: co-located {count} frames ({count / shorter_life:.0%} of shorter track)"))
    return pairs


def _box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Separation between two boxes: 0 when they touch or overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _lockstep_pairs(tracks: Dict[int, RawTrack], cfg: IdentityConfig) -> List[Tuple[int, int, str]]:
    """Find tracks rigidly attached to one another over many frames.

    This is the fix for the failure mode that defeated every earlier
    attempt: one large, close vehicle detected as several boxes TILED
    front-to-back along its body. Those boxes have low mutual IoU *and*
    low IoM -- no per-frame overlap threshold can merge them -- so the
    only honest evidence is temporal. Panels bolted to one bus keep a
    constant centre-to-centre offset frame after frame; two vehicles in
    adjacent lanes, even at similar speeds, do not hold that offset to
    within a few percent of a vehicle diagonal for a dozen frames.
    """
    if not cfg.lockstep_enabled:
        return []

    boxes_by_track: Dict[int, Dict[int, List[float]]] = {
        tid: {o.frame_idx: o.bbox for o in t.observations} for tid, t in tracks.items()
    }
    ids = sorted(tracks)
    pairs = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ta, tb = ids[i], ids[j]
            shared = sorted(set(boxes_by_track[ta]) & set(boxes_by_track[tb]))
            if len(shared) < cfg.lockstep_min_frames:
                continue

            offsets, gaps, diags = [], [], []
            for f in shared:
                ba, bb = boxes_by_track[ta][f], boxes_by_track[tb][f]
                ca, cb = _centre(ba), _centre(bb)
                offsets.append((cb[0] - ca[0], cb[1] - ca[1]))
                diag = max(_diag(ba), _diag(bb))
                diags.append(diag)
                smaller = min(_diag(ba), _diag(bb))
                gaps.append(_box_gap(ba, bb) / smaller if smaller > 0 else float("inf"))

            # Contiguous: panels of one body touch or overlap.
            if sum(g <= cfg.lockstep_max_gap_frac for g in gaps) / len(gaps) < 0.8:
                continue

            mean_diag = sum(diags) / len(diags)
            if mean_diag <= 0:
                continue
            xs = [o[0] for o in offsets]
            ys = [o[1] for o in offsets]
            drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / mean_diag
            if drift > cfg.lockstep_max_offset_drift_frac:
                continue

            pairs.append((
                ta, tb,
                f"lockstep: rigidly co-moving with track {tb} for {len(shared)} frames "
                f"(offset drift {drift:.1%} of body size)",
            ))
    return pairs


def _temporal_pairs(tracks: Dict[int, RawTrack], cfg: IdentityConfig) -> List[Tuple[int, int, str]]:
    """Find (dying track, newborn track) pairs that are one vehicle."""
    if not cfg.temporal_enabled:
        return []

    ordered = sorted(tracks.values(), key=lambda t: t.first_frame)
    pairs = []

    for new in ordered:
        best: Optional[Tuple[float, int, str]] = None
        for old in ordered:
            if old.track_id == new.track_id:
                continue
            gap = new.first_frame - old.last_frame
            # Strictly sequential: the old track must have ENDED before the
            # new one began. Overlapping tracks are the spatial case above,
            # and merging them here would hide genuine simultaneous vehicles.
            if gap < 0 or gap > cfg.temporal_max_gap_frames:
                continue

            old_box, new_box = old.observations[-1].bbox, new.observations[0].bbox

            size_ratio = _diag(new_box) / _diag(old_box) if _diag(old_box) > 0 else float("inf")
            if not (1 / cfg.temporal_max_size_ratio <= size_ratio <= cfg.temporal_max_size_ratio):
                continue

            # Where should the old track BE by now, coasting at its last
            # known velocity? A continuation lands near that prediction.
            vx, vy = old.velocity()
            oc = _centre(old_box)
            predicted = (oc[0] + vx * gap, oc[1] + vy * gap)
            nc = _centre(new_box)
            dist = math.hypot(nc[0] - predicted[0], nc[1] - predicted[1])
            # Tolerance scales with the vehicle's own size and grows with
            # the gap (the prediction gets less certain), but is capped --
            # a flat per-frame pixel budget becomes frame-sized on long gaps.
            body = max(_diag(old_box), _diag(new_box))
            allowed = body * min(
                cfg.temporal_tolerance_max_diag_frac,
                cfg.temporal_tolerance_diag_frac + cfg.temporal_tolerance_growth_per_frame * gap,
            )
            if dist > allowed:
                continue

            # Appearance veto: geometry cannot tell two vehicles following
            # the same lane at the same speed apart, but colour can.
            sim = appearance_similarity(old.appearance, new.appearance)
            if sim is not None and sim < cfg.temporal_min_appearance:
                continue

            # A near-stationary track has no meaningful heading, so the
            # direction gate is only applied when both tracks are moving.
            old_speed = math.hypot(vx, vy)
            nvx, nvy = new.head_velocity()
            new_speed = math.hypot(nvx, nvy)
            if old_speed > 1.0 and new_speed > 1.0:
                cos = (vx * nvx + vy * nvy) / (old_speed * new_speed)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                if angle > cfg.temporal_max_direction_deg:
                    continue

            # For an instant re-appearance (gap ~0) also require real box
            # overlap -- cheap protection against stitching to a different
            # vehicle following in the same lane at the same speed.
            if gap <= 2 and iou(old_box, new_box) < cfg.temporal_min_iou_at_gap:
                continue

            score = dist + gap * 2.0
            if best is None or score < best[0]:
                best = (
                    score,
                    old.track_id,
                    f"temporal: resumes track {old.track_id} after {gap}f gap, "
                    f"{dist:.0f}px from prediction (tolerance {allowed:.0f}px"
                    + (f", appearance {sim:.2f})" if sim is not None else ", no appearance)"),
                )
        if best is not None:
            pairs.append((best[1], new.track_id, best[2]))
    return pairs


def resolve_identities(
    tracks: Dict[int, RawTrack], cfg: IdentityConfig
) -> Tuple[List[VehicleIdentity], Dict[str, int]]:
    """Collapse raw tracks into physical vehicle identities."""
    stats = {"raw_tracks": len(tracks), "spatial_merges": 0, "lockstep_merges": 0, "temporal_merges": 0}
    if not tracks:
        return [], stats

    ds = _DisjointSet(tracks.keys())
    reasons: Dict[int, List[str]] = defaultdict(list)

    if cfg.enabled:
        for ta, tb, why in _spatial_pairs(tracks, cfg):
            if ds.union(ta, tb):
                stats["spatial_merges"] += 1
                reasons[ds.find(ta)].append(why)
        for ta, tb, why in _lockstep_pairs(tracks, cfg):
            if ds.union(ta, tb):
                stats["lockstep_merges"] += 1
                reasons[ds.find(ta)].append(why)
        for ta, tb, why in _temporal_pairs(tracks, cfg):
            if ds.union(ta, tb):
                stats["temporal_merges"] += 1
                reasons[ds.find(ta)].append(why)

    identities = []
    groups = sorted(ds.groups().values(), key=lambda g: min(tracks[t].first_frame for t in g))
    for vid, members in enumerate(groups, start=1):
        merged: List[Observation] = []
        for tid in members:
            merged.extend(tracks[tid].observations)
        merged.sort(key=lambda o: o.frame_idx)

        # A merged identity can hold several boxes on the same frame (the
        # spatial case). Keep one observation per frame -- the union extent
        # of that frame's boxes, so the crop still covers the whole vehicle.
        deduped: List[Observation] = []
        for obs in merged:
            if deduped and deduped[-1].frame_idx == obs.frame_idx:
                prev = deduped[-1]
                prev.bbox = [
                    min(prev.bbox[0], obs.bbox[0]),
                    min(prev.bbox[1], obs.bbox[1]),
                    max(prev.bbox[2], obs.bbox[2]),
                    max(prev.bbox[3], obs.bbox[3]),
                ]
                prev.conf = max(prev.conf, obs.conf)
            else:
                deduped.append(Observation(obs.frame_idx, list(obs.bbox), obs.conf))

        root = ds.find(members[0])
        identities.append(
            VehicleIdentity(
                vehicle_id=vid,
                raw_track_ids=sorted(members),
                observations=deduped,
                merge_reasons=reasons.get(root, []),
            )
        )

    stats["vehicles"] = len(identities)
    logger.info(
        "Identity resolution: %d raw tracks -> %d vehicles "
        "(%d spatial, %d lockstep, %d temporal merges)",
        stats["raw_tracks"],
        stats["vehicles"],
        stats["spatial_merges"],
        stats["lockstep_merges"],
        stats["temporal_merges"],
    )
    return identities, stats

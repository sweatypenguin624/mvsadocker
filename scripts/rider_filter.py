"""Rider/vehicle association filtering.

YOLO's class-0 "person" detections fire on people riding motorcycles and
bicycles as readily as on pedestrians on foot. This module estimates, on a
best-effort basis, whether a detected person is likely a rider rather than a
pedestrian, by measuring bbox overlap between the person and nearby vehicle
detections in the same frame.

This is explicitly NOT a reliable rider classifier. It is a geometric
heuristic and is reported as such: outputs are "pedestrian" / "rider" /
"uncertain", never a hard guarantee. It is disabled by default
(``rider_filter.enabled: false`` in config.yaml) so it can be validated
independently before being trusted to affect counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

from config_loader import RiderFilterConfig
from models import BBox, Detection

STATUS_PEDESTRIAN = "pedestrian"
STATUS_RIDER = "rider"
STATUS_UNCERTAIN = "uncertain"

VEHICLE_CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class RiderFilterResult:
    status: str
    matched_vehicle_class: Optional[str]
    overlap_ratio: float


def _bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


class RiderFilter:
    """Classifies a person detection as pedestrian/rider/uncertain based on
    overlap with vehicle-class detections in the same frame.
    """

    def __init__(self, config: RiderFilterConfig):
        self.config = config

    def classify(self, person_bbox: BBox, frame_detections: Sequence[Detection]) -> RiderFilterResult:
        """``frame_detections`` should be all detections (any class) found
        in the same frame as ``person_bbox``, typically including vehicle
        classes requested via ``rider_filter.vehicle_class_ids``.
        """
        person_area = _bbox_area(person_bbox)
        if person_area <= 0:
            return RiderFilterResult(STATUS_UNCERTAIN, None, 0.0)

        best_ratio = 0.0
        best_cls: Optional[int] = None
        for det in frame_detections:
            if det.cls_id not in self.config.vehicle_class_ids:
                continue
            overlap = _intersection_area(person_bbox, det.bbox)
            ratio = overlap / person_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_cls = det.cls_id

        if best_cls is None or best_ratio < self.config.uncertain_overlap_ratio:
            return RiderFilterResult(STATUS_PEDESTRIAN, None, best_ratio)

        vehicle_name = VEHICLE_CLASS_NAMES.get(best_cls, str(best_cls))
        if best_ratio >= self.config.rider_overlap_ratio:
            return RiderFilterResult(STATUS_RIDER, vehicle_name, best_ratio)
        return RiderFilterResult(STATUS_UNCERTAIN, vehicle_name, best_ratio)


def aggregate_track_rider_status(
    per_frame_statuses: List[str],
    per_frame_ratios: Optional[List[float]] = None,
    min_frames_for_confident_pedestrian: int = 1,
    short_track_max_clean_ratio: float = 0.0,
) -> str:
    """Majority-vote a track's per-frame rider statuses into one final
    status. "uncertain" votes are ignored unless there is no other signal,
    so a handful of ambiguous frames don't drown out a clear majority.

    A majority vote is only as reliable as the number of frames behind it.
    A track seen for only 1-2 frames has no averaging to fall back on: a
    single frame where the vehicle-overlap ratio happened to land just under
    ``uncertain_overlap_ratio`` -- most often because the rider's own body
    was occluding their vehicle from the detector, e.g. mid-mount/dismount
    -- becomes the track's entire verdict. Confirmed against
    peds_18_kamaripet_police_station_cam_5_20260728_1208: several short
    tracks with a lone "pedestrian" vote were, on inspection of
    tracked.mp4, riders caught mid-transition, not pedestrians.

    So for a track shorter than ``min_frames_for_confident_pedestrian``
    whose vote comes out "pedestrian", also require every one of its
    per-frame overlap ratios to be at or below ``short_track_max_clean_ratio``
    (i.e. no vehicle was ever measurably close) before trusting it --
    otherwise downgrade to "uncertain" rather than counting it as a
    confident pedestrian. Longer tracks are left to the plain majority vote,
    since enough frames average out any single occlusion-related miss.
    """
    if not per_frame_statuses:
        return STATUS_UNCERTAIN
    decisive = [s for s in per_frame_statuses if s != STATUS_UNCERTAIN]
    pool = decisive if decisive else per_frame_statuses
    counts = Counter(pool)
    verdict = counts.most_common(1)[0][0]

    if (
        verdict == STATUS_PEDESTRIAN
        and per_frame_ratios
        and len(per_frame_statuses) < min_frames_for_confident_pedestrian
        and max(per_frame_ratios) > short_track_max_clean_ratio
    ):
        return STATUS_UNCERTAIN

    return verdict
